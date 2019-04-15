import os

from PIL import Image
import numpy as np
from transformation import *
from api_utils import detect_safe_search, detect_label, localize_objects
import cv2
from image_utils import draw_boxes

img = cv2.imread('images/0000.png')

''' Exampls using the transformation API '''

''' Example code to generate adv_example/affin.png '''
# img = affine_transform(img, 15, bg_color=(0,0,0))

''' Example code to generate adv_example/perspective.png '''
img = perspective_transform(img, coordinates=[[50, 50], [530, 50], [50, 550], [430, 430]])

''' Example code to generate adv_example/color_filtering.png '''
# img = color_filtering(img)

''' Example code to generate adv_example/perspective.png '''
# img = blur(img, (5,5))

#cv2.imwrite('layers_car/large_6.jpg', img)
#objects = localize_objects('layers_car/large_12.jpg')
ret = detect_safe_search('images/0000.png')
print(ret)
'''
for i in range(0, 2000):
    ret = detect_safe_search('images/0000.png')
    print(ret)
'''
'''
labels = []
coords = []
image = Image.open("layers_car/large_12.jpg")
for object in objects:
    label = object.name + ":" + str(object.score)[:4]
    labels.append(label)
    top = object.bounding_poly.normalized_vertices[0].y
    left = object.bounding_poly.normalized_vertices[0].x
    bottom = object.bounding_poly.normalized_vertices[2].y
    right = object.bounding_poly.normalized_vertices[2].x
    coords.append([top, left, bottom, right])

draw_boxes(image, labels, coords)
image.save("images/out.png", "PNG")
'''
