import os
import sys
import shutil
import torchvision
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from defense.preprocessor import DIM_tranform
from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb  




def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating preprocessed images from AEs.')
    parser.add_argument('AE_folder', help='AE folder path.', type=str)
    parser.add_argument('--preprocess-mtd',  help='Preprocessing method.', default='DIM', type=str)
    parser.add_argument('--output-dir',  help='Output folder.', default='/home/yantao/temp', type=str)

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    folder_name = args.AE_folder.split('/')[-1] + '_preprocess_' + args.preprocess_mtd
    output_folder = os.path.join(args.output_dir, folder_name)
    if os.path.exists(output_folder):
        raise ValueError('output folder {0} existed.'.format(output_folder))
    os.mkdir(output_folder)
    img_list = os.listdir(args.AE_folder)

    if args.preprocess_mtd == 'DIM':
        preprocessor = DIM_tranform(224, 224, 330)

    for img_name in tqdm(img_list):
        image_np = load_image(shape=(224, 224), data_format='channels_first', abs_path=True, fpath=os.path.join(args.AE_folder, img_name))
        image_var = numpy_to_variable(image_np)
        image_processed_var = preprocessor(image_var)
        image_processed_np = image_processed_var[0].detach().cpu().numpy()
        image_processed_pil = Image.fromarray((np.transpose(image_processed_np, (1, 2, 0)) * 255).astype(np.uint8))
        image_processed_pil.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":
    main()
