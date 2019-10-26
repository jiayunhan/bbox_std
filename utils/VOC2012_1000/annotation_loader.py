import os
import numpy as np
from six import raise_from
from PIL import Image

import pdb

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    

voc_classes = {
    'aeroplane'   : 0,
    'bicycle'     : 1,
    'bird'        : 2,
    'boat'        : 3,
    'bottle'      : 4,
    'bus'         : 5,
    'car'         : 6,
    'cat'         : 7,
    'chair'       : 8,
    'cow'         : 9,
    'diningtable' : 10,
    'dog'         : 11,
    'horse'       : 12,
    'motorbike'   : 13,
    'person'      : 14,
    'pottedplant' : 15,
    'sheep'       : 16,
    'sofa'        : 17,
    'train'       : 18,
    'tvmonitor'   : 19
}


def name_to_label(name):
        """ Map name to label.
        """
        return voc_classes[name]

def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result

def __parse_annotation(element, ratio):
    """ Parse an annotation given an XML element.
    """
    truncated = _findNode(element, 'truncated', parse=int)
    difficult = _findNode(element, 'difficult', parse=int)

    class_name = _findNode(element, 'name').text

    box = np.zeros((4,))
    label = name_to_label(class_name)

    bndbox    = _findNode(element, 'bndbox')
    box[0] = float(_findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1) * ratio[0]
    box[1] = float(_findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1) * ratio[1]
    box[2] = float(_findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1) * ratio[0]
    box[3] = float(_findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1) * ratio[1]

    return truncated, difficult, box, label

def __parse_annotations(xml_root, img_size):
    """ Parse all annotations under the xml_root.
    """
    gt_w = int(xml_root.find('size').find('width').text)
    gt_h = int(xml_root.find('size').find('height').text)
    ratio = (float(img_size[0]) / float(gt_h), float(img_size[1]) / float(gt_w))

    annotations = {'labels': np.empty((len(xml_root.findall('object')),)), 'bboxes': np.empty((len(xml_root.findall('object')), 4))}
    for i, element in enumerate(xml_root.iter('object')):
        try:
            truncated, difficult, box, label = __parse_annotation(element, ratio)
        except ValueError as e:
            raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

        annotations['bboxes'][i, :] = box
        annotations['labels'][i] = label

    return annotations

def load_annotations(file_path, img_size):
    """ Load annotations for an image_index.
    """
    try:
        tree = ET.parse(file_path)
        return __parse_annotations(tree.getroot(), img_size)
    except ET.ParseError as e:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
    except ValueError as e:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)