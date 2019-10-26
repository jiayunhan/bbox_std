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
        category_index={1: {'id': 1, 'name': 'person'},
                        2: {'id': 2, 'name': 'bicycle'},
                        3: {'id': 3, 'name': 'car'},
                        4: {'id': 4, 'name': 'motorcycle'},
                        5: {'id': 5, 'name': 'airplane'},
                        6: {'id': 6, 'name': 'bus'},
                        7: {'id': 7, 'name': 'train'},
                        8: {'id': 8, 'name': 'truck'},
                        9: {'id': 9, 'name': 'boat'},
                        10: {'id': 10, 'name': 'traffic light'},
                        11: {'id': 11, 'name': 'fire hydrant'},
                        13: {'id': 13, 'name': 'stop sign'},
                        14: {'id': 14, 'name': 'parking meter'}}  

        coco_91class = {
            0 : '__background__',
            1 : 'person',
            2 : 'bicycle',
            3 : 'car',
            4 : 'motorcycle',
            5 : 'airplane',
            6 : 'bus',
            7 : 'train',
            8 : 'truck',
            9 : 'boat',
            10 : 'traffic light',
            11 : 'fire hydrant',
            12 : 'street sign N/A',
            13 : 'stop sign',
            14 : 'parking meter',
            15 : 'bench',
            16 : 'bird',
            17 : 'cat',
            18 : 'dog',
            19 : 'horse',
            20 : 'sheep',
            21 : 'cow',
            22 : 'elephant',
            23 : 'bear',
            24 : 'zebra',
            25 : 'giraffe',
            26 : 'hat N/A',
            27 : 'backpack',
            28 : 'umbrella',
            29 : 'shoe N/A',
            30 : 'eye glasses N/A',
            31 : 'handbag',
            32 : 'tie',
            33 : 'suitcase',
            34 : 'frisbee',
            35 : 'skis',
            36 : 'snowboard',
            37 : 'sports ball',
            38 : 'kite',
            39 : 'baseball bat',
            40 : 'baseball glove',
            41 : 'skateboard',
            42 : 'surfboard',
            43 : 'tennis racket',
            44 : 'bottle',
            45 : 'plate N/A',
            46 : 'wine glass',
            47 : 'cup',
            48 : 'fork',
            49 : 'knife',
            50 : 'spoon',
            51 : 'bowl',
            52 : 'banana',
            53 : 'apple',
            54 : 'sandwich',
            55 : 'orange',
            56 : 'broccoli',
            57 : 'carrot',
            58 : 'hot dog',
            59 : 'pizza',
            60 : 'donut',
            61 : 'cake',
            62 : 'chair',
            63 : 'couch',
            64 : 'potted plant',
            65 : 'bed',
            66 : 'mirror N/A',
            67 : 'dining table',
            68 : 'window N/A',
            69 : 'desk N/A',
            70 : 'toilet',
            71 : 'door N/A',
            72 : 'tv',
            73 : 'laptop',
            74 : 'mouse',
            75 : 'remote',
            76 : 'keyboard',
            77 : 'cell phone',
            78 : 'microwave',
            79 : 'oven',
            80 : 'toaster',
            81 : 'sink',
            82 : 'refrigerator',
            83 : 'blender N/A',
            84 : 'book',
            85 : 'clock',
            86 : 'vase',
            87 : 'scissors',
            88 : 'teddy bear',
            89 : 'hair drier',
            90 : 'toothbrush'
        }
        
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
                    box_lurd = np.array([temp_box_resize[0], temp_box_resize[1], temp_box_resize[2], temp_box_resize[3]])
                    temp_result['bbox'] = box_lurd
                    temp_result['score'] = temp_score
                    temp_result['class_idx'] = temp_class
                    try:
                        temp_result['class_name'] = category_index[temp_class]['name']
                    except:
                        temp_result['class_name'] = 'None'
                    results.append(temp_result)

        return results