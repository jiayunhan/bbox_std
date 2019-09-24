import os
import sys
import shutil
import torchvision
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from attacks.dispersion import DispersionAttack_gpu
from models.vgg import Vgg16
from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb                       


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)
    parser.add_argument('--adv-method',  help='Adversarial attack method.', default='dr', type=str)
    parser.add_argument('--target-model',  help='Target model for generating AEs.', default='vgg16', type=str)
    parser.add_argument('--epsilon', help='Budget for attack.', default=16, type=int)
    parser.add_argument('--step-size', help='Step size in range of 0 - 255', default=1, type=int)
    parser.add_argument('--steps', help='Number of steps.', default=2000, type=int)

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    with open('utils/labels.txt','r') as inf:
        args_dic['imagenet_dict'] = eval(inf.read())

    args_dic['input_dir'] = os.path.join(args.dataset_dir, 'ori')

    target_model = None
    internal = None
    attack = None
    attack_layer_idx = None

    if args.target_model == 'vgg16':
        target_model = Vgg16()
        internal = [i for i in range(29)]
        attack_layer_idx = 14
    
    if args.adv_method == 'dr':
        attack = DispersionAttack_gpu(target_model, epsilon=args.epsilon/255., step_size=args.step_size/255., steps=args.steps)

    assert target_model != None and internal != None and attack != None and attack_layer_idx != None

    args_dic['output_dir'] = os.path.join(
        args.dataset_dir, 
        '{0}_{1}_layerAt_{2:02d}_eps_{3}_stepsize_{4}_steps_{5}'.format(
            args.adv_method, 
            args.target_model, 
            attack_layer_idx, 
            args.epsilon,
            args.step_size,
            args.steps,
        )
    )
    
    if os.path.exists(args.output_dir):
        raise ValueError('Output folder existed.')
    os.mkdir(args.output_dir)

    for image_name in tqdm(os.listdir(args.input_dir)):
        image_path = os.path.join(args.input_dir, image_name)

        image_np = load_image(data_format='channels_first', abs_path=True, fpath=image_path)
        image_var = numpy_to_variable(image_np)

        adv = attack(
            image_var, 
            attack_layer_idx=attack_layer_idx,
            internal=internal
        )

        adv_np = variable_to_numpy(adv)
        image_pil = Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0)))
        image_pil.save(os.path.join(args.output_dir, os.path.splitext(image_name)[0] + '.png'))


if __name__ == '__main__':
    main()
    