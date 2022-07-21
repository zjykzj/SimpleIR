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
from zcls2.util import logging

logger = logging.get_logger(__name__)

from simpleir.configs import get_cfg_defaults
from simpleir.engine.infer import validate
from simpleir.data.build import build_data
from simpleir.models.build import build_model
from simpleir.utils.util import load_model


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


def main():
    global best_prec_list, best_epoch, args

    args = parse()
    cfg = init_cfg(args)

    top_k = cfg.TRAIN.TOP_K
    best_prec_list = [0 for _ in top_k]
    best_epoch = 0

    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(cfg, device)

    # Optionally resume from a checkpoint
    if cfg.RESUME:
        logger.info("=> Resume now")
        load_model(model, cfg.RESUME, device=device)

    # Data loading code
    _, query_loader = build_data(cfg, is_train=False, is_gallery=False, w_path=True)
    _, gallery_loader = build_data(cfg, is_train=False, is_gallery=True, w_path=True)

    start = time.time()
    logger.info("=> Evaluate now")

    validate(cfg, model, query_loader, gallery_loader, device=device)

    end = time.time()
    logger.info("One epoch eval need: {:.3f}".format((end - start)))


if __name__ == '__main__':
    main()
