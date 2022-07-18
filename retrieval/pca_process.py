# -*- coding: utf-8 -*-

"""
@date: 2022/7/17 下午5:29
@file: pca_process.py
@author: zj
@description: Load the data set, train the PCA model parameters, and save the PCA model
"""
import os
import glob
import joblib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sknormalize
from argparse import ArgumentParser, RawTextHelpFormatter

PCA_WAYS = ['pca', 'pca_w']


def parse_args():
    parser = ArgumentParser(description="Compute PCA/Whitening", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--feature-dir', metavar="FEATURE", default=None, type=str,
                        help='Dir for loading features. Default: None')

    parser.add_argument('--way', metavar='WAY', default='pca', type=str, choices=PCA_WAYS,
                        help='The way to reduce dimension. Default: pca')
    parser.add_argument('--reduce-dimension', metavar='DIMENSION', default=512, type=int,
                        help='Dimension after dimension reduction. Default: 512')

    parser.add_argument('--save-dir', metavar='SAVE', default=None,
                        help='Dir for saving PCA model. Default: None')
    return parser.parse_args()


def load_features(feature_dir):
    assert os.path.isdir(feature_dir), feature_dir

    feat_file_list = glob.glob(os.path.join(feature_dir, "*.npy"))

    features = []
    names = []
    for feat_path in feat_file_list:
        feat_name = os.path.splitext(os.path.basename(feat_path))[0]
        feat = np.load(feat_path)

        names.append(feat_name)
        features.append(feat.reshape(-1))

    return features, names


def normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
    else:
        return sknormalize(x, copy=copy)


def fit(features, rd=512, is_whiten=False):
    """
    Calculate pca/whitening parameters
    """
    # Normalize
    features = normalize(features)

    # Whiten and reduce dimension
    pca = PCA(n_components=rd, whiten=is_whiten)
    pca.fit(features)

    return pca


def main():
    args = parse_args()
    print('args:', args)

    print('Loading features %s ...' % str(args.feature_dir))
    data, _ = load_features(args.feature_dir)

    print('Fitting PCA/whitening wth d=%d on %s ...' % args.reduce_dimension)
    model = fit(data, args.reduce_dimension, is_whiten=args.way == 'pca_w')

    print('Saving PCA model to %s ...' % args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    res_path = os.path.join(args.save_dir, 'pca.gz')
    joblib.dump(model, res_path)


if __name__ == '__main__':
    main()
