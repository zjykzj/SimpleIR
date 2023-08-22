# -*- coding: utf-8 -*-

"""
@date: 2022/10/30 上午11:13
@file: make_roxford5k_rparis6k.py
@author: zj
@description: Downloading Oxford5k/Paris6k/ROxford5k/RParis6k
@See:
1. https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
2. https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

Refer to
1. https://github.com/filipradenovic/revisitop/blob/master/python/download.py
2. https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/download.py
"""

import os
import argparse

DATA_LIST = ['oxford5k', 'paris6k']


def parse_args():
    parser = argparse.ArgumentParser(description="Downloading Oxford5k/Paris6k")
    parser.add_argument('--dataset', metavar='DATASET', default="oxford5k", type=str,
                        choices=DATA_LIST,
                        help='Dataset to be downloaded. Default: oxford5k')
    parser.add_argument('--root', metavar='ROOT', default="./data", type=str,
                        help='Path of dataset to be saved. Default: ./data')

    return parser.parse_args()


def download(data_dir, dataset):
    assert os.path.isdir(data_dir), data_dir
    assert dataset in DATA_LIST

    if dataset == 'oxford5k':
        remote_dir = 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
        dl_files = ['oxbuild_images-v1.tgz']
        gnd_dl_files = ['gt_files_170407.tgz']
    elif dataset == 'paris6k':
        remote_dir = 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
        dl_files = ['paris_1-v1.tgz', 'paris_2-v1.tgz']
        gnd_dl_files = ['paris_120310.tgz']
    else:
        raise ValueError('ERROR')

    dst_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for dl_name in dl_files:
        remote_url = os.path.join(remote_dir, dl_name)
        dst_path = os.path.join(data_dir, dl_name)

        print('>> Downloading dataset {} archive {}...'.format(dataset, dl_name))
        os.system('wget {} -O {}'.format(remote_url, dst_path))

        print('>> Extracting dataset {} archive {}...'.format(dataset, dl_name))
        # create tmp folder
        dst_dir_tmp = os.path.join(data_dir, 'tmp')
        os.system('mkdir {}'.format(dst_dir_tmp))
        # extract in tmp folder
        os.system('tar -zxf {} -C {}'.format(dst_path, dst_dir_tmp))
        # remove all (possible) subfolders by moving only files in dst_dir
        os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
        # remove tmp folder
        os.system('rm -rf {}'.format(dst_dir_tmp))

        print('>> Extracted, deleting dataset {} archive {}...'.format(dataset, dst_path))
        os.system('rm {}'.format(dst_path))

    for gnd_dl_name in gnd_dl_files:
        gnd_remote_url = os.path.join(remote_dir, gnd_dl_name)

        gnd_dst_file = os.path.join(data_dir, gnd_dl_name)
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} ground truth file...'.format(dataset))
            os.system('wget {} -O {}'.format(gnd_remote_url, gnd_dst_file))
            print('>> Extracting dataset {} archive {}...'.format(dataset, gnd_dl_name))
            # create folder
            dst_dir = os.path.join(data_dir, 'groundtruth')
            os.system('mkdir {}'.format(dst_dir))
            # extract in folder
            os.system('tar -zxf {} -C {}'.format(gnd_dst_file, dst_dir))
            # remove gnd_file
            os.system('rm -rf {}'.format(gnd_dst_file))


def main(args):
    dataset = args.dataset.lower()
    data_root = args.root

    data_dir = os.path.join(data_root, dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download(data_dir, dataset)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    main(args)
