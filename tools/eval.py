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

from zcls2.util.distributed import init_dist
from zcls2.util.parser import parse, load_cfg
from zcls2.util.collect_env import collect_env_info
from zcls2.util.misc import resume

from zcls2.util import logging

logger = logging.get_logger(__name__)

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

    logger.info("CUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    logger.info("Loaded configuration file: {}".format(args.config))
    if args.config:
        cfg.merge_from_file(args.config)
        with open(args.config, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Loaded args: {args}")
    load_cfg(args, cfg)

    logger.info("Running with config:\n{}".format(cfg))

    return cfg


def load_model(model, model_path, device=torch.device('cpu')):
    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError(f'There is no key `state_dict` in {model_path}')

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logger.info("=> loaded checkpoint '{}'".format(model_path, ))


def main():
    global best_prec_list, best_epoch, args

    args = parse()
    cfg = init_cfg(args)

    top_k = cfg.TRAIN.TOP_K
    best_prec_list = [0 for _ in top_k]
    best_epoch = 0

    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(cfg, device)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(cfg).to(device)

    # Optionally resume from a checkpoint
    if cfg.RESUME:
        logger.info("=> Resume now")
        load_model(model, cfg.RESUME, device=device)

    # # Data loading code
    _, _, val_loader = build_data(cfg)

    if cfg.EVALUATE:
        logger.info("=> Evaluate now")
        validate(cfg, val_loader, model, criterion)
        return


if __name__ == '__main__':
    main()
