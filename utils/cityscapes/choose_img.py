import argparse
import sys
import warnings
import os
import shutil
import random
from PIL import Image
import numpy as np

import pdb
    

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for choosing image from CityScapes val set.')
    parser.add_argument('--folder', help='Imagenet validation folder path.', default='/home/yantao/workspace/cityscapes/leftImg8bit/val', type=str)
    parser.add_argument('--output-dir',  help='Directory for saved images.', default='/home/yantao/img_output_cityscapes', type=str)
    parser.add_argument('--num-imgs',  help='Number of images to choose.', default=500, type=int)
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if os.path.exists(args.output_dir):
        raise ValueError('Output folder existed.')
    os.mkdir(args.output_dir)

    folder_names = os.listdir(args.folder)
    img_paths = []
    for temp_folder_name in folder_names:
        name_file_path = os.path.join(args.folder, temp_folder_name)
        img_names = os.listdir(name_file_path)
        for img_name in img_names:
            img_paths.append(os.path.join(name_file_path, img_name))

    num_files = len(img_paths)
    assert num_files >= args.num_imgs
    choices = np.random.choice(img_paths, args.num_imgs, replace=False)
    for curt_choice in choices:
        curt_name = curt_choice.split('/')[-1]
        shutil.copy(curt_choice, os.path.join(args.output_dir, curt_name))

if __name__ == "__main__":
    main()