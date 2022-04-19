# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: main.py
@author: zj
@description: 
"""

from datetime import datetime
import argparse
import torch
import torch.nn as nn

from simpleir.models.tiny_autocoder import TinyAutoCoder
from simpleir.data.build import build_dataloader
from simpleir.engine.trainer import train_epoch
from simpleir.utils.util import save_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (default: true)')
    return parser.parse_args()


def process(gpu, args):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = TinyAutoCoder(in_channels=1).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Data loading code
    train_data_loader = build_dataloader(args, train=True)

    dst_root = './outputs/'

    start = datetime.now()
    for epoch in range(args.epochs):
        epoch_start = datetime.now()
        train_epoch(epoch, args, model, device, train_data_loader, optimizer, criterion)
        print("Training one epoch in: " + str(datetime.now() - epoch_start))
        lr_scheduler.step()

        save_model(model.cpu(), dst_root=dst_root, epoch=epoch)

    print("Training complete in: " + str(datetime.now() - start))


def main():
    args = parse_args()
    process(0, args)


if __name__ == '__main__':
    main()
