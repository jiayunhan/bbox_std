from pycocotools.coco import COCO
import os
import numpy as np
import cv2
import pickle

import pdb

def load_masks(img_name_noext, img_size, coco_ann, to_voc=None):
    """ Load annotations for an image_index.
    """

    ori_h = coco_ann.loadImgs(ids=[int(img_name_noext)])[0]['height']
    ori_w = coco_ann.loadImgs(ids=[int(img_name_noext)])[0]['width']
    ratio = (float(img_size[0]) / float(ori_h), float(img_size[1]) / float(ori_w))
    annotations_ids = coco_ann.getAnnIds(imgIds=[int(img_name_noext)])
    mask_output = np.zeros(img_size).astype(np.int)
    if len(annotations_ids) == 0:
        return mask_output
    coco_annotations = coco_ann.loadAnns(annotations_ids)
    for idx, curt_ann in enumerate(coco_annotations):
        # some annotations have basically no width / height, skip them
        if curt_ann['bbox'][2] < 1 or curt_ann['bbox'][3] < 1:
            continue
        curt_mask = coco_ann.annToMask(curt_ann)
        curt_id = curt_ann['category_id']
        curt_mask_resize = _resize_mask(curt_mask, img_size)
        if to_voc != None:
            if curt_id in to_voc.keys():
                locs = np.argwhere(curt_mask_resize == 1)
                for loc in locs:
                    mask_output[loc[0], loc[1]] = to_voc[curt_id]

    return mask_output

def _resize_mask(ori_mask, output_size):
    resized = cv2.resize(ori_mask, output_size, interpolation=cv2.INTER_NEAREST)
    return resized