# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:33
@file: trainer.py
@author: zj
@description: 
"""

import time

from typing import Optional
from yacs.config import CfgNode

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from zcls2.config.key_word import KEY_OUTPUT
from zcls2.util.meter import AverageMeter
from zcls2.util.prefetcher import data_prefetcher
from zcls2.util.metric import accuracy
from zcls2.util.distributed import reduce_tensor
from zcls2.util.misc import to_python_float
from zcls2.optim.lr_scheduler.build import adjust_learning_rate
from zcls2.util import logging

logger = logging.get_logger(__name__)


def train(cfg: CfgNode, train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer,
          epoch: Optional[int] = 1) -> None:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    warmup = cfg.LR_SCHEDULER.IS_WARMUP
    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH

    prefetcher = data_prefetcher(cfg, train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        if cfg.PROF >= 0 and i == cfg.PROF:
            logger.info("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if cfg.PROF >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        if warmup and epoch < warmup_epoch + 1:
            adjust_learning_rate(cfg, optimizer, epoch, i, len(train_loader))

        # compute output
        if cfg.PROF >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if cfg.PROF >= 0: torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if cfg.PROF >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if cfg.PROF >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if cfg.PROF >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if cfg.PROF >= 0: torch.cuda.nvtx.range_pop()

        if i % cfg.PRINT_FREQ == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if cfg.DISTRIBUTED:
                reduced_loss = reduce_tensor(cfg.NUM_GPUS, loss.data)
            else:
                reduced_loss = loss.data

            if cfg.TRAIN.CALCULATE_ACCURACY:
                # Measure accuracy
                prec1, prec5 = accuracy(output[KEY_OUTPUT].data, target, topk=(1, 5))

                # Average loss and accuracy across processes for logging
                if cfg.DISTRIBUTED:
                    prec1 = reduce_tensor(cfg.NUM_GPUS, prec1)
                    prec5 = reduce_tensor(cfg.NUM_GPUS, prec5)
            else:
                prec1, prec5 = [0.], [0.]

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / cfg.PRINT_FREQ)
            end = time.time()

            if cfg.RANK_ID == 0:
                logger.info('Epoch: [{0}/{1}][{2}/{3}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Speed {4:.3f} ({5:.3f}) '
                            'Lr {lr:.10f} '
                            'Loss {loss.val:.10f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, cfg.TRAIN.MAX_EPOCH, i, len(train_loader),
                    cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.val,
                    cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.avg,
                    batch_time=batch_time,
                    lr=optimizer.param_groups[0]['lr'],
                    loss=losses, top1=top1, top5=top5))
        if cfg.PROF >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        input, target = prefetcher.next()
        if cfg.PROF >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if cfg.PROF >= 0: torch.cuda.nvtx.range_pop()

        if cfg.PROF >= 0 and i == cfg.PROF + 10:
            logger.info("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
