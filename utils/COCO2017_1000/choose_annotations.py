import os
import shutil

import pdb

ori_dir = '/home/yantao/workspace/datasets/COCO2017_1000/ori'
ann_dir = '/home/yantao/_annotations'
out_dir = '/home/yantao/workspace/datasets/COCO2017_1000/_annotations'

os.mkdir(out_dir)
for temp_name in os.listdir(ori_dir):
    temp_ann_name = os.path.splitext(temp_name)[0] + '.xml'
    shutil.copy(os.path.join(ann_dir, temp_ann_name), os.path.join(out_dir, temp_ann_name))

