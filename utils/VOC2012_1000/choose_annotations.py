import os
import shutil

import pdb

ori_dir = '/home/yantao/workspace/datasets/VOC2012_1000_gt/ori'
ann_dir = '/home/yantao/_annotations'
ann_out_dir = '/home/yantao/workspace/datasets/VOC2012_1000_gt/_annotations'
seg_dir = '/home/yantao/_segmentations'
seg_out_dir = '/home/yantao/workspace/datasets/VOC2012_1000_gt/_segmentations'


os.mkdir(ann_out_dir)
for temp_name in os.listdir(ori_dir):
    temp_ann_name = os.path.splitext(temp_name)[0] + '.xml'
    shutil.copy(os.path.join(ann_dir, temp_ann_name), os.path.join(ann_out_dir, temp_ann_name))

os.mkdir(seg_out_dir)
for temp_name in os.listdir(ori_dir):
    temp_seg_name = os.path.splitext(temp_name)[0] + '.png'
    shutil.copy(os.path.join(seg_dir, temp_seg_name), os.path.join(seg_out_dir, temp_seg_name))

