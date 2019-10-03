import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
from glob import glob

import pdb

class SSD_detector(object):
    def __init__(self):
        
        detect_model_name = 'models/ssd_mobilenet/ssd_mobilenet_v1_coco_11_06_2017'
        
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)      

    def predict(self, image_pil):
        '''
        return dictionary of list

        Output:
        {
            'boxes' : [[top, left, bottom, right], ...]
            'scores' : [float, ...]
            'classes' : [int, ...]
            'class_names' : [str, ...]
        }
        '''
        image_np = np.array(image_pil).astype(float)
        det_res = self.detect_image(image_np) 
        ret_dic = {
            'boxes' : [],
            'scores' : [],
            'classes' : [],
            'class_names' : []
        }
        for temp_det in det_res:
            ret_dic['boxes'].append(temp_det['bbox'].tolist())
            ret_dic['scores'].append(temp_det['score'])
            ret_dic['classes'].append(temp_det['class_idx'])
            ret_dic['class_names'].append(temp_det['class_name'])
        return ret_dic
        
    def detect_image(self, image, th_conf=0.3):  
        
        """Determines the locations of the cars in the image

        Args:
            image: numpy array

        Returns:
        detected objects with: bbox, confident score, class index
        [
            dictionary {
                bbox: np.array([left, up, right, down])
                score: confident_score
                class_idx: class_idx
                class_name: class name category
            }
        ]

        """
        category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorcycle'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}  
        
        results = []

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
            
            boxes=np.squeeze(boxes)
            classes =np.squeeze(classes)
            scores = np.squeeze(scores)
            for temp_box, temp_score, temp_class in zip(boxes, scores, classes):
                temp_class = int(temp_class)
                dim = np.array(image.shape[0:2])
                temp_result = {}
                if temp_score > th_conf:
                    temp_box_resize = self.box_normal_to_pixel(temp_box, dim)
                    box_lurd = np.array([temp_box_resize[1], temp_box_resize[0], temp_box_resize[3], temp_box_resize[2]])
                    temp_result['bbox'] = box_lurd
                    temp_result['score'] = temp_score
                    temp_result['class_idx'] = temp_class
                    try:
                        temp_result['class_name'] = category_index[temp_class]['name']
                    except:
                        temp_result['class_name'] = 'None'
                    results.append(temp_result)

        return results