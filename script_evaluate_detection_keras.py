import os
import sys
import shutil
import json
from keras import backend as K
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from models.yolov3.yolov3_wrapper import YOLOv3
from models.retina_resnet50.keras_retina_resnet50 import KerasResNet50RetinaNetModel
from models.ssd_mobilenet.SSD import SSD_detector
from utils.image_utils import load_image, save_image, save_bbox_img
from utils.mAP import save_detection_to_file, calculate_mAP_from_files

import pdb                       


# python script_evaluate_detection_keras.py yolov3 --dataset-dir /home/yantao/workspace/datasets/VOC2012_1000


PICK_LIST = []
BAN_LIST = []

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

    if args.test_model == 'yolov3':
        test_model = YOLOv3(sess = K.get_session())
        img_size = (416, 416)
    elif args.test_model == 'retina_resnet50':
        test_model = KerasResNet50RetinaNetModel()
        img_size = (416, 416)
    elif args.test_model == 'ssd_mobile':
        test_model = SSD_detector()
        img_size = (500, 500)

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

        result_dir = 'temp_dect_results'
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'gt'))
        os.mkdir(os.path.join(result_dir, 'pd'))

        for image_name in tqdm(os.listdir(input_dir)):
            temp_image_name_noext = os.path.splitext(image_name)[0]
            ori_img_path = os.path.join(input_dir, image_name)
            adv_img_path = os.path.join(args.dataset_dir, curt_folder, image_name)
            adv_img_path = os.path.splitext(adv_img_path)[0] + '.png'
            if not os.path.exists(adv_img_path):
                print('File {0} not found.'.format(image_name))
                continue
            
            image_ori_np = load_image(data_format='channels_last', shape=img_size, bounds=(0, 255), abs_path=True, fpath=ori_img_path)
            Image.fromarray((image_ori_np).astype(np.uint8)).save(os.path.join(result_dir, 'ori.jpg'))
            image_ori_pil = Image.fromarray(image_ori_np.astype(np.uint8))
            gt_out = test_model.predict(image_ori_pil)
            
            image_adv_np = load_image(data_format='channels_last', shape=img_size, bounds=(0, 255), abs_path=True, fpath=adv_img_path)
            Image.fromarray((image_adv_np).astype(np.uint8)).save(os.path.join(result_dir, 'temp_adv.jpg'))
            image_adv_pil = Image.fromarray(image_adv_np.astype(np.uint8))
            pd_out = test_model.predict(image_adv_pil)

            save_detection_to_file(gt_out, os.path.join(result_dir, 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
            save_detection_to_file(pd_out, os.path.join(result_dir, 'pd', temp_image_name_noext + '.txt'), 'detection')
            
            
            if gt_out:
                save_bbox_img(os.path.join(result_dir, 'ori.jpg'), gt_out['boxes'], out_file='temp_ori_box.jpg')
            else:
                save_bbox_img(os.path.join(result_dir, 'ori.jpg'), [], out_file='temp_ori_box.jpg')
            if pd_out:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.jpg'), pd_out['boxes'], out_file='temp_adv_box.jpg')
            else:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.jpg'), [], out_file='temp_adv_box.jpg')
            

        mAP_score = calculate_mAP_from_files(os.path.join(result_dir, 'gt'), os.path.join(result_dir, 'pd'))
        shutil.rmtree(result_dir)
        print(curt_folder, ' : ', mAP_score)
        result_dict[curt_folder] = str(mAP_score)

        with open('temp_det_results_{0}.json'.format(args.test_model), 'w') as fout:
            json.dump(result_dict, fout, indent=2)


if __name__ == '__main__':
    main()
