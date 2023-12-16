# -*- coding:utf-8 -*-
"""
@Project: NCL
@File：organize_val_folders.py
@Author：Sihang Xie
@Time：2023/12/16 11:00
@Description：Reorganize the ImageNet validation images into their respective category folders.
"""

import os
import shutil
import json
from scipy.io import loadmat
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Reorganize the ImageNet validation images into their respective category folders')

    parser.add_argument(
        "--val_path",
        help="Path to the original ImageNet validation images",
        required=False,
        type=str,
        default="/home/og/XieSH/dataset/long-tailed/public/ImageNet_LT/val"
    )

    parser.add_argument(
        "--devkit_path",
        help="Path to the ILSVRC2012_validation_ground_truth.txt",
        required=False,
        type=str,
        default="/home/og/XieSH/dataset/long-tailed/public/ImageNet_LT/ILSVRC2012_devkit_t12"
    )

    args = parser.parse_args()
    return args


def organize_val_folders(args):
    val_path = args.val_path
    devkit_path = args.devkit_path

    # load meta data of validation set
    try:
        synset = loadmat(os.path.join(devkit_path, 'data', 'meta.mat'))
        ground_truth = open(os.path.join(devkit_path, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    except Exception as e:
        print(str(e))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    # get all file names of validation images
    root, _, filenames = next(os.walk(val_path))

    for filename in filenames:
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id: %d, ILSVRC_ID: %d, WIND: %s " % (val_id, ILSVRC_ID, WIND))

        # move validation images
        output_path = os.path.join(root, WIND)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.move(os.path.join(root, filename), os.path.join(output_path, filename))


if __name__ == '__main__':
    args = parse_args()
    organize_val_folders(args)
