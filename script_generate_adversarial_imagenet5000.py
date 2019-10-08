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
from attacks.DIM import DIM_Attack
from attacks.mifgsm import MomentumIteratorAttack
from attacks.linf_pgd import LinfPGDAttack
from models.vgg import Vgg16
from models.resnet import Resnet152
from models.inception import Inception_v3
from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb                       

# DR        : python script_generate_adversarial_imagenet5000.py
# DIM       : python script_generate_adversarial_imagenet5000.py --adv-method dim --step-size 25.5 --steps 40
# mi-FGSM   : python script_generate_adversarial_imagenet5000.py --adv-method mifgsm --step-size 25.5 --steps 40
# PGD       : python script_generate_adversarial_imagenet5000.py --adv-method pgd --step-size 25.5 --steps 40

# python script_generate_adversarial_imagenet5000.py -tm inception_v3 --step-size 4 --steps 100

DEBUG = False

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)
    parser.add_argument('--adv-method',  help='Adversarial attack method.', default='dr', type=str)
    parser.add_argument('--loss-method',  help='Loss function for DR attack.', default='std', type=str)
    parser.add_argument('-tm', '--target-model',  help='Target model for generating AEs.', default='vgg16', type=str)
    parser.add_argument('--epsilon', help='Budget for attack.', default=16, type=int)
    parser.add_argument('--step-size', help='Step size in range of 0 - 255', default=1, type=float)
    parser.add_argument('--steps', help='Number of steps.', default=2000, type=int)
    parser.add_argument('--inc3-attacklayer', help='Inception v3 attack layer idx.', default=-1, type=int)

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
    if args.adv_method == 'dr':
        loss_mtd = args.loss_method
        if args.target_model == 'vgg16':
            target_model = Vgg16()
            internal = [i for i in range(29)]
            attack_layer_idx = [14] # 12, 14
            args_dic['image_size'] = (224, 224)
        elif args.target_model == 'resnet152':
            target_model = Resnet152()
            internal = [i for i in range(9)]
            attack_layer_idx = [8]
            args_dic['image_size'] = (224, 224)
        elif args.target_model == 'inception_v3':
            assert args.inc3_attacklayer != -1
            target_model = Inception_v3()
            internal = [i for i in range(14)]
            attack_layer_idx = [args.inc3_attacklayer]
            args_dic['image_size'] = (299, 299)
        else:
            raise

        attack = DispersionAttack_gpu(
            target_model, 
            epsilon=args.epsilon/255., 
            step_size=args.step_size/255., 
            steps=args.steps, 
            loss_mtd=loss_mtd
        )

    elif args.adv_method == 'dim' or args.adv_method == 'mifgsm' or args.adv_method == 'pgd':
        attack_layer_idx = [0]
        internal = [0]
        loss_mtd = ''

        if args.target_model == 'vgg16':
            target_model = torchvision.models.vgg16(pretrained=True).cuda().eval()
            args_dic['image_size'] = (224, 224)
        elif args.target_model == 'resnet152':
            target_model = torchvision.models.resnet152(pretrained=True).cuda().eval()
            args_dic['image_size'] = (224, 224)
        elif args.target_model == 'inception_v3':
            target_model = torchvision.models.inception_v3(pretrained=True).cuda().eval()
            args_dic['image_size'] = (299, 299)
        else:
            raise ValueError('Invalid adv_method.')

        if args.adv_method == 'dim':
            attack = DIM_Attack(
                target_model, 
                decay_factor=1, 
                prob=0.5, 
                epsilon=args.epsilon/255., 
                step_size=args.step_size/255., 
                steps=args.steps, 
                image_resize=330
            )
        elif args.adv_method == 'mifgsm':
            attack = MomentumIteratorAttack(
                target_model,
                decay_factor=0.5, 
                epsilon=args.epsilon/255., 
                step_size=args.step_size/255., 
                steps=args.steps, 
                random_start=False
            )
        elif args.adv_method == 'pgd':
            attack = LinfPGDAttack(
                target_model, 
                epsilon=args.epsilon/255., 
                a=args.step_size/255., 
                k=args.steps,  
                random_start=False
            )
        
    else:
        raise ValueError('Invalid adv_mdthod.')
    assert target_model != None and internal != None and attack != None and attack_layer_idx != None
    attack_layer_idx_str = ''
    for layer_idx in attack_layer_idx:
        attack_layer_idx_str += (str(layer_idx) + '_')
    attack_layer_idx_str = attack_layer_idx_str[:-1]

    if not DEBUG:
        args_dic['output_dir'] = os.path.join(
            args.dataset_dir, 
            '{0}_{1}_layerAt_{2}_eps_{3}_stepsize_{4}_steps_{5}_lossmtd_{6}'.format(
                args.adv_method, 
                args.target_model, 
                attack_layer_idx_str, 
                args.epsilon,
                args.step_size,
                args.steps,
                loss_mtd
            )
        )
        if os.path.exists(args.output_dir):
            raise ValueError('Output folder existed.')
        os.mkdir(args.output_dir)

    for image_name in tqdm(os.listdir(args.input_dir)):
        image_path = os.path.join(args.input_dir, image_name)

        image_np = load_image(shape=args.image_size, data_format='channels_first', abs_path=True, fpath=image_path)
        image_var = numpy_to_variable(image_np)
        if args.adv_method == 'dr':
            adv = attack(
                image_var,
                attack_layer_idx,
                internal
            )
        else:
            target_model.eval()
            logits_nat = target_model(image_var)
            y_var = logits_nat.argmax().long().unsqueeze(0)
            adv = attack(
                image_var.cpu(), 
                y_var.cpu()
            )

        if not DEBUG:
            adv_np = variable_to_numpy(adv)
            image_pil = Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0)))
            image_pil.save(os.path.join(args.output_dir, os.path.splitext(image_name)[0] + '.png'))


if __name__ == '__main__':
    main()
    