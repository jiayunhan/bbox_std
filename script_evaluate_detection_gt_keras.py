import os
import sys
import shutil
import json
from keras import backend as K
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from pycocotools.coco import COCO

from models.yolov3.yolov3_wrapper import YOLOv3
from models.retina_resnet50.keras_retina_resnet50 import KerasResNet50RetinaNetModel
from models.retina_resnet50.retinanet_resnet_50.utils.image import preprocess_image
from models.ssd_mobilenet.SSD import SSD_detector
from utils.image_utils import load_image, save_image, save_bbox_img
from utils.mAP import save_detection_to_file, calculate_mAP_from_files
from utils.VOC2012_1000.annotation_loader import load_annotations as load_voc_annotations
from utils.COCO2017_1000.annotation_loader import load_annotations as load_coco_annotations

import pdb                       


# python script_evaluate_detection_gt_keras.py yolov3 coco --dataset-dir /home/yantao/workspace/datasets/VOC2012_1000


# {voc_idx : coco_idx}
with open('utils/VOC_AND_COCO91_CLASSES.pkl', 'rb') as f:
    VOC_AND_COCO91_CLASSES = pickle.load(f)
with open('utils/VOC_AND_COCO80_CLASSES.pkl', 'rb') as f:
    VOC_AND_COCO80_CLASSES = pickle.load(f)

PICK_LIST = [
    'ori'
]
BAN_LIST = []

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('test_model', help='Model for testing AEs.', type=str)
    parser.add_argument('dataset_type', choices=['coco', 'voc'], help='Dataset for testing AEs.', type=str)
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    if args.dataset_type == 'voc':
        gt_dir = os.path.join(args.dataset_dir, '_annotations')
    elif args.dataset_type == 'coco':
        gt_loader = COCO(os.path.join(args.dataset_dir, 'instances_val2017.json'))

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
        if temp_folder == 'imagenet_val_5000' or temp_folder == '.git' or temp_folder == '_annotations' or temp_folder == '_segmentations':
            continue 
        if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
            continue
        if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
            continue
        test_folders.append(temp_folder)

    result_dict = {}
    for curt_folder in tqdm(test_folders):
        print('Folder : {0}'.format(curt_folder))

        result_dir = 'temp_dect_results'
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'gt'))
        os.mkdir(os.path.join(result_dir, 'pd'))

        for adv_name in tqdm(os.listdir(os.path.join(args.dataset_dir, curt_folder))):
            temp_image_name_noext = os.path.splitext(adv_name)[0]
            if args.dataset_type == 'voc':
                gt_path = os.path.join(gt_dir, temp_image_name_noext + '.xml')

            if curt_folder == 'ori':
                adv_img_path = os.path.join(args.dataset_dir, curt_folder, temp_image_name_noext + '.jpg')
            else:
                adv_img_path = os.path.join(args.dataset_dir, curt_folder, temp_image_name_noext + '.png')

            if not os.path.exists(adv_img_path):
                print('File {0} not found.'.format(adv_img_path))
                continue
            
            if args.dataset_type == 'voc':
                gt_out = load_voc_annotations(gt_path, img_size)
                gt_out['classes'] = gt_out['classes'].astype(np.int)
            elif args.dataset_type == 'coco':
                gt_out = load_coco_annotations(temp_image_name_noext, img_size, gt_loader)
            
            image_adv_np = load_image(data_format='channels_last', shape=img_size, bounds=(0, 255), abs_path=True, fpath=adv_img_path)
            Image.fromarray((image_adv_np).astype(np.uint8)).save(os.path.join(result_dir, 'temp_adv.jpg'))
            image_adv_pil = Image.fromarray(image_adv_np.astype(np.uint8))
            pd_out = test_model.predict(image_adv_pil)
            if args.dataset_type == 'voc':
                pd_out = _transfer_label_to_voc(pd_out, args)
            elif args.dataset_type == 'coco':
                if args.test_model == 'yolov3' or args.test_model == 'retina_resnet50':
                    pd_out = _transfer_label_to_coco91(pd_out, args)

            save_detection_to_file(gt_out, os.path.join(result_dir, 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
            save_detection_to_file(pd_out, os.path.join(result_dir, 'pd', temp_image_name_noext + '.txt'), 'detection')
            
            if pd_out:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.jpg'), pd_out['boxes'], out_file='temp_adv_box.jpg')
            else:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.jpg'), [], out_file='temp_adv_box.jpg')
            

        mAP_score = calculate_mAP_from_files(os.path.join(result_dir, 'gt'), os.path.join(result_dir, 'pd'))

        shutil.rmtree(result_dir)
        print(curt_folder, ' : ', mAP_score)
        result_dict[curt_folder] = 'mAP: {0:.04f}'.format(mAP_score)

        with open('temp_det_results_{0}.json'.format(args.test_model), 'w') as fout:
            json.dump(result_dict, fout, indent=2)

def _transfer_label_to_voc(pd_out, args):
    if args.test_model == 'ssd_mobile':
        voc_and_coco_classes = VOC_AND_COCO91_CLASSES
    elif args.test_model == 'yolov3' or args.test_model == 'retina_resnet50':
        voc_and_coco_classes = VOC_AND_COCO80_CLASSES

    ret = {
        'classes' : [],
        'scores' : [],
        'boxes' : []
    }
    for key in pd_out.keys():
        if key not in ret.keys():
            ret[key] = pd_out[key]
    classes_list = pd_out['classes']
    scores_list = pd_out['scores']
    boxes_list = pd_out['boxes']
    for idx, temp_class in enumerate(classes_list):
        for key, val in voc_and_coco_classes.items():
            if int(temp_class) == val:
                ret['classes'].append(int(key))
                ret['scores'].append(scores_list[idx])
                ret['boxes'].append(boxes_list[idx])
    return ret

def _transfer_label_to_coco91(pd_out, args):
    milestones = [0, 11, 24, 26, 40, 60, 61, 62, 73]
    add_list = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    classes_list = pd_out['classes']
    for class_idx, temp_class in enumerate(classes_list):
        add_idx = 0
        while add_idx < len(milestones):
            if add_idx == len(milestones) - 1:
                break
            if milestones[add_idx + 1] > temp_class:
                break
            add_idx += 1
        pd_out['classes'][class_idx] = temp_class + add_list[add_idx]
    return pd_out

if __name__ == '__main__':
    main()
