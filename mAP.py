import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np

import pdb

def save_detection_to_file(input_dic, output_file, task):
    with open(output_file, 'w') as txt_file:
        if not input_dic:
            return
        class_list = input_dic['classes']
        bbox_list = input_dic['boxes']
        scores_list = input_dic['scores']
        for temp_class, temp_bbox, temp_score in zip(class_list, bbox_list, scores_list):
            top, left, bottom, right = temp_bbox
            if task == 'ground_truth':
                txt_file.write(temp_class + "," + str(int(left)) + "," + str(int(top)) + "," + str(int(right)) + "," + str(int(bottom)) + '\n')
            elif task == 'detection':
                txt_file.write(temp_class + "," + str(temp_score) + "," + str(int(left)) + "," + str(int(top)) + "," + str(int(right)) + "," + str(int(bottom)) + '\n')
            else:
                raise ValueError('Invalid task.')

def calculate_mAP_from_files(gt_dir, pd_dir, min_overlap=0.5):
    """
    Create a ".temp_files/" and "results/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)

    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = _file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        already_seen_classes = []
        for line in lines_list:
            class_name, left, top, right, bottom = line.split(',')
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name" : class_name, "bbox" : bbox, "used" : False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    # get a list with the detection-results files
    dr_files_list = glob.glob(pd_dir + '/*.txt')
    dr_files_list.sort()

    for txt_file_pd in dr_files_list:
        lines = _file_lines_to_list(txt_file_pd)
        for line in lines:
            tmp_class_name, confidence, left, top, right, bottom = line.split(',')
            if tmp_class_name not in list(gt_counter_per_class.keys()):
                gt_counter_per_class[tmp_class_name] = 0
    
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    for _, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = _file_lines_to_list(txt_file)
            for line in lines:
                tmp_class_name, confidence, left, top, right, bottom = line.split(',')
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence" : confidence, "file_id" : file_id, "bbox" : bbox})
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x : float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    # open file to store the results
    count_true_positives = {}
    for _, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
        Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
        Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            # set minimum overlap
            min_overlap = min_overlap
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            if gt_counter_per_class[class_name] == 0:
                rec[idx] = 0
            else:
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, _, _ = voc_ap(rec[:], prec[:])
        sum_AP += ap

    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    print(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)
    return mAP

def _file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


if __name__ == "__main__":
    calculate_mAP_from_files('out/DispersionAttack_opt_det_out/gt', 'out/DispersionAttack_opt_det_out/pd')