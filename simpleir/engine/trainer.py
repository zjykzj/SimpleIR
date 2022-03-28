# -*- coding: utf-8 -*-

"""
@date: 2022/2/9 下午9:07
@file: engine.py
@author: zj
@description: 
"""

import os
import torch


def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):
    model.train()
    model = model.to(device)
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        # loss = criterion(output, target.to(device))
        loss = criterion(data.to(device), output)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('[PID {}]\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader), optimizer.state_dict()['param_groups'][0]['lr'],
                loss.item()))
            if args.dry_run:
                break
