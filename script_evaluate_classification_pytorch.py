import os
import sys
import shutil
import torchvision
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import pickle

from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb                       

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('test_model', help='Model for testing AEs.', type=str)
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    with open('utils/labels.txt','r') as inf:
        args_dic['imagenet_dict'] = eval(inf.read())

    input_dir = os.path.join(args.dataset_dir, 'ori')

    if args.test_model == 'resnet50':
        test_model = torchvision.models.resnet50(pretrained=True).cuda()
        test_model.eval()

    test_folders = []
    for temp_folder in os.listdir(args.dataset_dir):
        if not os.path.isdir(os.path.join(args.dataset_dir, temp_folder)):
            continue 
        if temp_folder == 'imagenet_val_5000' or temp_folder == 'ori' or temp_folder == '.git':
            continue 
        test_folders.append(temp_folder)
    
    result_dict = {}
    for curt_folder in test_folders:
        print('Folder : {0}'.format(curt_folder))
        correct_count = 0
        total_count = 0
        for image_name in tqdm(os.listdir(input_dir)):
            image_ori_path = os.path.join(input_dir, image_name)
            image_adv_path = os.path.join(args.dataset_dir, curt_folder, image_name)
            image_adv_path = os.path.splitext(image_adv_path)[0] + '.png'
            image_ori_np = load_image(data_format='channels_first', abs_path=True, fpath=image_ori_path)
            image_adv_np = load_image(data_format='channels_first', abs_path=True, fpath=image_adv_path)
            image_ori_var = numpy_to_variable(image_ori_np)
            image_adv_var = numpy_to_variable(image_adv_np)
            
            logits_ori = test_model(image_ori_var)
            logits_adv = test_model(image_adv_var)

            y_ori_var = logits_ori.argmax()
            y_adv_var = logits_adv.argmax()

            total_count += 1
            if y_ori_var == y_adv_var:
                correct_count += 1
        print('{0} samples are correctly labeled over {1} samples.'.format(correct_count, total_count))
        acc = float(correct_count) / float(total_count)
        print('Accuracy for {0} : {1}'.format(curt_folder, acc))
        result_dict[curt_folder] = str(acc)

    with open('temp_cls_results_{0}.pkl'.format(args.test_model), 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    main()
    