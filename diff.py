from numpy import linalg as LA
from PIL import Image
import cv2
import numpy as np
#im1 = Image.open("images/origin.jpg")
im1 = cv2.imread("images/0000.png")
im2 = cv2.imread("layers_car/large_12.jpg")
diff = im1 - im2
a = LA.norm(diff[0], np.inf)
print(a)
