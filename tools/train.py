import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from yacs.config import CfgNode
from argparse import Namespace

from zcls2.data.transform.build import create_mixup_fn
from zcls2.data.dataset.mp_dataset import MPDataset
from zcls2.optim.optimizer.build import build_optimizer
from zcls2.optim.lr_scheduler.build import build_lr_scheduler
from zcls2.engine.trainer import train
from zcls2.util.distributed import init_dist
from zcls2.util.parser import parse, load_cfg
from zcls2.util.collect_env import collect_env_info
from zcls2.util.checkpoint import save_checkpoint
from zcls2.util.misc import resume

from zcls2.util import logging

logger = logging.get_logger(__name__)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from simpleir.configs import get_cfg_defaults
from simpleir.engine.infer import validate
from simpleir.data.build import build_data
from simpleir.models.build import build_model
from simpleir.criterion.build import build_criterion


def init_cfg(args: Namespace) -> CfgNode:
    cfg = get_cfg_defaults()
    init_dist(args, cfg)

    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)
    if args.local_rank == 0 and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    logging.setup_logging(local_rank=args.local_rank, output_dir=cfg.OUTPUT_DIR)
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("local_rank: {0}, master_addr: {1}, master_port: {2}".format(
        os.environ['LOCAL_RANK'], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
    logger.info("opt_level = {}".format(args.opt_level))
    logger.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32, type(args.keep_batchnorm_fp32)))
    logger.info("loss_scale = {}".format(args.loss_scale, type(args.loss_scale)))

    logger.info("CUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    logger.info("Loaded configuration file: {}".format(args.config))
    if args.config:
        cfg.merge_from_file(args.config)
        with open(args.config, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Loaded args: {args}")
    load_cfg(args, cfg)

    # Scale learning rate based on global batch size
    cfg.OPTIMIZER.LR = cfg.OPTIMIZER.LR * float(cfg.DATALOADER.TRAIN_BATCH_SIZE * cfg.NUM_GPUS) / 256.

    logger.info("Running with config:\n{}".format(cfg))

    return cfg


def main():
    global best_prec_list, best_epoch, args

    args = parse()
    cfg = init_cfg(args)

    top_k = cfg.TRAIN.TOP_K
    best_prec_list = [0 for _ in top_k]
    best_epoch = 0

    device = torch.device(f'cuda:{cfg.RANK_ID}') if cfg.DISTRIBUTED else torch.device('cpu')
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if cfg.DISTRIBUTED:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(cfg).to(device)

    # Optionally resume from a checkpoint
    if cfg.RESUME:
        logger.info("=> Resume now")
        resume(cfg, model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device)

    # Data loader
    train_sampler, train_loader = build_data(cfg, is_train=True, w_path=False)
    _, query_loader = build_data(cfg, is_train=False, is_gallery=False, w_path=True)
    _, gallery_loader = build_data(cfg, is_train=False, is_gallery=True, w_path=True)
    mixup_fn = create_mixup_fn(cfg)

    if cfg.EVALUATE:
        logger.info("=> Evaluate now")
        validate(cfg, model, query_loader, gallery_loader, device=device)
        return

    warmup = cfg.LR_SCHEDULER.IS_WARMUP
    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH

    torch.distributed.barrier()
    assert cfg.TRAIN.START_EPOCH > 0
    logger.info("=> Train now")
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH + 1):
        # train for one epoch
        start = time.time()

        if isinstance(train_loader.dataset, MPDataset):
            train_loader.dataset.set_epoch(epoch)
        elif cfg.DISTRIBUTED:
            assert isinstance(train_sampler, torch.utils.data.DistributedSampler)
            train_sampler.set_epoch(epoch)

        train(cfg, train_loader, model, criterion, optimizer, epoch=epoch, mixup_fn=mixup_fn)
        torch.cuda.empty_cache()
        if warmup and epoch < (warmup_epoch + 1):
            pass
        else:
            lr_scheduler.step()

        end = time.time()
        logger.info("One epoch train need: {:.3f}".format((end - start)))

        if epoch % cfg.TRAIN.EVAL_EPOCH == 0:
            # evaluate on validation set
            start = time.time()
            # prec1, prec5 = validate(cfg, val_loader, model, criterion)
            prec_list = validate(cfg, model, query_loader, gallery_loader, device=device)
            torch.cuda.empty_cache()

            is_best = prec_list[0] > best_prec_list[0]
            if is_best:
                best_prec_list = prec_list
                best_epoch = epoch

            logger_str = f' BestEpoch: [{best_epoch}]'
            logger.info(logger_str)
            logger_str = ' * '
            for k, prec in zip(top_k, best_prec_list):
                logger_str += f'Prec@{k} {prec:.3f} '
            logger.info(logger_str)

            # remember best prec@1 and save checkpoint
            if cfg.RANK_ID == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model.state_dict(),
                    'prec_list': prec_list,
                    'best_prec_list': best_prec_list,
                    'best_epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, is_best, output_dir=cfg.OUTPUT_DIR, filename=f'checkpoint_{epoch}.pth.tar')

            end = time.time()
            logger.info("One epoch validate need: {:.3f}".format((end - start)))
        torch.distributed.barrier()


if __name__ == '__main__':
    main()
