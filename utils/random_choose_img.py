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
    parser = argparse.ArgumentParser(description='Script for choosing image from folder.')
    parser.add_argument('folder', help='Folder path.', type=str)
    parser.add_argument('--output-dir',  help='Directory for saved images.', default='/home/yantao/img_output', type=str)
    parser.add_argument('--num-imgs',  help='Number of images to choose.', default=1000, type=int)
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if os.path.exists(args.output_dir):
        raise ValueError('Output folder {0} exisited.'format(args.output_dir))
    os.mkdir(args.output_dir)

    name_list = os.listdir(args.folder)
    img_name_list = []
    for temp_name in name_list:
        temp_ext = os.path.splitext(temp_name)[-1]
        if temp_ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            img_name_list.append(temp_name)
    num_files = len(img_name_list)
    assert num_files >= args.num_imgs
    choices = np.random.choice(img_name_list, args.num_imgs, replace=False)
    for curt_choice in choices:
        curt_path = os.path.join(args.folder, curt_choice)
        shutil.copy(curt_path, os.path.join(args.output_dir, curt_choice))
    

if __name__ == '__main__':
    main()
    