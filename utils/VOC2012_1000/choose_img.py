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
    parser = argparse.ArgumentParser(description='Script for choosing image from VOC2012 val set.')
    parser.add_argument('--folder', help='Imagenet validation folder path.', default='/home/yantao/workspace/VOC2012/VOCdevkit/VOC2012', type=str)
    parser.add_argument('--name-file',  help='Name list file in /ImageSets/Main/.', default='val.txt', type=str)
    parser.add_argument('--output-dir',  help='Directory for saved images.', default='/home/yantao/VOC2012_1000_output', type=str)
    parser.add_argument('--num-imgs',  help='Number of images to choose.', default=1000, type=int)
    parser.add_argument('--disable-random',     help='Disable random choosing.', dest='random_choice', action='store_false')
    parser.add_argument('--file-list', default=None, type=list, help='file list to choose image.')
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    
    if args.file_list is not None:
        warnings.warn('file_list is used, only listed images are chosen.')
        raise NotImplementedError

    name_file_path = os.path.join(args.folder, 'ImageSets', 'Main', args.name_file)
    with open(name_file_path, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)
        assert num_files >= args.num_imgs
        choices = np.random.choice(lines, args.num_imgs, replace=False)
        for curt_choice in choices:
            curt_path = os.path.join(args.folder, 'JPEGImages', curt_choice.strip() + '.jpg')
            shutil.copy(curt_path, os.path.join(args.output_dir, curt_choice.strip() + '.jpg'))
    

if __name__ == '__main__':
    main()
    