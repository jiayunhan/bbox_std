import argparse
import sys
import warnings
import os
import shutil
import random
from PIL import Image
import numpy as np

import torch
import torchvision

import pdb


def load_folder_index():
    '''load file 'folder_name_to_index.txt'.
    '''
    folder_label_dic = {}
    with open('./folder_name_to_label.txt') as f:
        lines = f.readlines()
        for line in lines:
            folder_name, index, label = line[:-1].split(' ')
            folder_label_dic[folder_name] = label.replace('_', ' ')
    index_label_dic = {}
    with open('./label_to_index.txt') as f:
        lines = f.readlines()
        for line in lines:
            curt_idx, curt_label = line.strip().split(': ')
            index_label_dic[int(curt_idx)] = curt_label
    folder_index_dic = {}
    for folder_name, label_name_1 in folder_label_dic.items():
        folder_index_dic[folder_name] = []
        is_found = False
        for index, label_name_2 in index_label_dic.items():
            if label_name_2.find(label_name_1) != -1:
                is_found = True
                folder_index_dic[folder_name].append(int(index))
        if not is_found:
            raise ValueError('labels in two files does not match.')
    return folder_index_dic
    

def choose_img(folder_name, folder_index, model, args):
    '''choose certain number of images given the images folder
    '''
    dst_folder_path = os.path.join(args.output_dir, folder_name)
    os.mkdir(dst_folder_path)

    src_folder_path = os.path.join(args.folder, folder_name)
    img_name_list = os.listdir(src_folder_path)
    
    if args.random_choice:
        random.shuffle(img_name_list)
    count = 0
    for curt_name in img_name_list:
        if count == args.num_per_class:
            break
        if not args.correct_label:
            shutil.copyfile(os.path.join(src_folder_path, curt_name), os.path.join(dst_folder_path, curt_name))
            count += 1
        else:
            assert model is not None, 'model should not ne None when --correct-label is True.'
            img_pil = Image.open(os.path.join(src_folder_path, curt_name)).convert('RGB')
            normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            loader = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                    normalize
                ]
            )
            img_t = loader(img_pil).unsqueeze_(0)
            #img_np = (img_t[0].detach().numpy().transpose((1, 2, 0))*255).astype(np.uint8)
            #Image.fromarray(img_np).save('temp.png')
            pred = model(img_t).detach().numpy()
            pred_label = np.argmax(pred[0])
            print('gt: {0} --- pd: {1}'.format(folder_index, pred_label))
            if pred_label in folder_index:
                shutil.copyfile(os.path.join(src_folder_path, curt_name), os.path.join(dst_folder_path, curt_name))
                count += 1
    if count < args.num_per_class:
        warnings.warn('Not enough correctly predicted images. Only {0} images collected.'.format(count))
    

def check_output(args):
    '''Validation check for generated images
    '''
    print('Checking...')
    assert os.path.exists(args.output_dir)
    folder_list = os.listdir(args.output_dir)
    assert len(folder_list) == 1000, 'The number of generated sub-folders should be 1000 instead of {0}'.format(len(folder_list))
    for folder_name in folder_list:
        img_list = os.listdir(os.path.join(args.output_dir, folder_name))
        if len(img_list) != args.num_per_class:
            warnings.warn('The number of images in {0} is {1} instead of {2}'.format(folder_name, len(img_list), args.num_per_class))
    print('Checking finished.')
    

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for choosing image from imagenet val set.')
    parser.add_argument('--folder',         help='Imagenet validation folder path.', default='/home/yantao/imagenet/imagenet_dataset/val', type=str)
    parser.add_argument('--output-dir',  help='Directory for saved images.', default='./output', type=str)
    parser.add_argument('--num-per-class',  help='Number of chosen images per class.', default=5, type=int)
    parser.add_argument('--disable-random',     help='Disable random choosing.', dest='random_choice', action='store_false')
    parser.add_argument('--file-list', default=None, type=list, help='file list to choose image.')
    parser.add_argument('--correct-label', help='Only choose correctly predicted images.', dest='correct_label', action='store_true')
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

    if args.correct_label:
        print('Only choose correctly predicted images based on Resnet50.')
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
    else:
        model = None

    folder_to_index = load_folder_index()
    for folder in folder_to_index.keys():
        choose_img(folder, folder_to_index[folder], model, args)

    check_output(args)
    

if __name__ == '__main__':
    main()
    