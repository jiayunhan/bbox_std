import torch
import sys
import argparse
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import json

from models.deeplabv3plus.utils.metrics import Evaluator
from models.deeplabv3plus.modeling.deeplab import *
from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb


PICK_LIST = [
    'tidim_vgg16_layerAt_00_eps_16_stepsize_3.2_steps_10',
]
BAN_LIST = []

def test(args):
    input_dir = os.path.join(args.dataset_dir, 'ori')

    model = DeepLab(
        num_classes=args.num_classes,
        backbone=args.backbone,
        output_stride=8,
        sync_bn=False,
        freeze_bn=True
    )

    model = model.cuda()
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    evaluator = Evaluator(args.num_classes)

    test_folders = []
    for temp_folder in os.listdir(args.dataset_dir):
        if not os.path.isdir(os.path.join(args.dataset_dir, temp_folder)):
            continue 
        if temp_folder == 'imagenet_val_5000' or temp_folder == 'ori' or temp_folder == '.git':
            continue 
        if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
            continue
        if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
            continue
        test_folders.append(temp_folder)
    
    result_dict = {}
    for curt_folder in test_folders:
        print('Folder : {0}'.format(curt_folder))
        evaluator.reset()
        for image_name in tqdm(os.listdir(input_dir)):
            temp_image_name_noext = os.path.splitext(image_name)[0]
            ori_img_path = os.path.join(input_dir, image_name)
            adv_img_path = os.path.join(args.dataset_dir, curt_folder, image_name)
            adv_img_path = os.path.splitext(adv_img_path)[0] + '.png'

            image_ori_np = load_image(
                data_format='channels_first', 
                shape=(args.crop_size, args.crop_size), 
                bounds=(0, 1), 
                abs_path=True, 
                fpath=ori_img_path
            )
            #Image.fromarray(np.transpose(image_ori_np * 255., (1, 2, 0)).astype(np.uint8)).save('ori.jpg')
            image_ori_var = numpy_to_variable(image_ori_np)
            with torch.no_grad():
                output_ori = model(image_ori_var)
            #Image.fromarray((output_ori[0].argmax(axis=0).cpu().numpy().astype(np.float32) / 21. * 255.).astype(np.uint8)).save('ori_fm.jpg')
            
            image_adv_np = load_image(
                data_format='channels_first', 
                shape=(args.crop_size, args.crop_size), 
                bounds=(0, 1), 
                abs_path=True, 
                fpath=adv_img_path
            )
            #Image.fromarray(np.transpose(image_adv_np * 255., (1, 2, 0)).astype(np.uint8)).save('temp_adv.jpg')
            image_adv_var = numpy_to_variable(image_adv_np)
            with torch.no_grad():
                output_adv = model(image_adv_var)
            #Image.fromarray((output_adv[0].argmax(axis=0).cpu().numpy().astype(np.float32) / 21. * 255.).astype(np.uint8)).save('adv_fm.jpg')

            pred_ori = output_ori.data.cpu().numpy()
            pred_ori = np.argmax(pred_ori, axis=1)
            pred_adv = output_adv.data.cpu().numpy()
            pred_adv = np.argmax(pred_adv, axis=1)

            evaluator.add_batch(pred_ori, pred_adv)

        try:
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        except:
            Acc = 0.
            Acc_class = 0.
            mIoU = 0.
            FWIoU = 0.

        result_str = 'Acc : {0:.04f}, Acc_class : {1:.04f}, mIoU : {2:.04f}, FWIoU : {3:.04f}'.format(Acc, Acc_class, mIoU, FWIoU)
        print(curt_folder, ' : ', result_str)
        result_dict[curt_folder] = result_str

    with open('temp_cls_results_{0}.json'.format(args.backbone), 'w') as fout:
        json.dump(result_dict, fout, indent=2)
        

def parse_args(args):
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('backbone', type=str,
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset-dir', help='Dataset folder path.', 
                        default='/home/yantao/workspace/datasets/imagenet5000', type=str)
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--num-classes', type=int, default=21,
                        help='number of classes')
    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)
    if args.backbone == 'resnet':
        args_dic['pretrained_path'] = 'models/deeplabv3plus/pretrained/deeplab-resnet.pth.tar'
    elif args.backbone == 'drn':
        args_dic['pretrained_path'] = 'models/deeplabv3plus/pretrained/deeplab-drn.pth.tar'
    elif args.backbone == 'mobile':
        args_dic['pretrained_path'] = 'models/deeplabv3plus/pretrained/deeplab-mobilenet.pth.tar'
    else:
        raise ValueError()

    test(args)

if __name__ == "__main__":
   main()