# -*- coding: utf-8 -*-

"""
@date: 2022/5/21 下午2:26
@file: prefetcher.py
@author: zj
@description: Custom Data Prefetcher. Derived from ZCls2
"""

import torch


class data_prefetcher():
    def __init__(self, cfg, loader):
        self.len = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        mean, std, _, max_value = cfg.TRANSFORM.NORMALIZE
        assert len(mean) == len(std)

        mean_list = list()
        std_list = list()
        for m, s in zip(mean, std):
            mean_list.append(m * max_value)
            std_list.append(s * max_value)
        n = len(mean_list)
        self.mean = torch.tensor(mean_list).cuda().view(1, n, 1, 1)
        self.std = torch.tensor(std_list).cuda().view(1, n, 1, 1)

        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_path = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.next_path
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        # return input, target
        return input, target, path

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        return self.next()
