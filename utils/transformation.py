import cv2
import numpy as np

def blur(image, ksize=(2,2)):
    """ 对图片进行模糊
        参数:
            image: 图片(ndarray)
            ksize: 模糊处理的kernel大小，越大模糊程度越高
    """
    return cv2.blur(image, ksize)

def affine_transform(image, angle, bg_color=(0,0,0), padding=(150, 150, 150, 150)):
    """ 仿射变化(类似平移旋转)
        参数:
            image: 图片(ndarray)
            angle: 旋转的角度
            bg_color: 旋转后补全用的背景颜色
            padding: (top, bottom, left, right) 为保证旋转后图片完整而加的 padding 大小
    """    
    top, bottom, left, right = padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=bg_color)
    h, w, _ = image.shape
    M = cv2.getRotationMatrix2D((h/2, w/2), angle, 1)
    image = cv2.warpAffine(image, M, (h, w), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=bg_color)
    return image

def perspective_transform(image, coordinates, bg_color=(0,0,0), padding=(0, 0, 0, 0)):
    """ 透视变换(将矩形转换至任意四边形形状)
        参数:
            image: 图片(ndarray)
            coordinates: 透视变换目标四边形的四个顶点坐标,如 [[0,0],[100,0],[0,100],[100,100]]
            bg_color: 变换后补全用的背景颜色
            padding: (top, bottom, left, right) 为保证变换后图片完整而加的 padding 大小
    """

    h, w, _ = image.shape
    top, bottom, left, right = padding
    new_h, new_w = h + top + bottom, w + left + right
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=bg_color)
    
    x0, y0, x1, y1 = top, left, top + h, left + w
    
    old_coordinates = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    new_coordinates = np.float32(coordinates)

    M = cv2.getPerspectiveTransform(old_coordinates, new_coordinates)
    image = cv2.warpPerspective(image, M, (new_h, new_w), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=bg_color)

    return image

def color_filtering(image, gbr_factor=(1., 0., 0.)):
    """ 颜色的偏色变换
        参数:
            image: 图片(ndarray)
            gbr_factor: G, B, R 三个颜色通道乘以的系数(0~1)，(1, 1, 1) 为维持原状， (0, 1, 0) 为仅保留绿色通道
    """
    image = image.astype(np.float)
    image[:,:,0] *= gbr_factor[0]
    image[:,:,1] *= gbr_factor[1]
    image[:,:,2] *= gbr_factor[2]
    image = image.astype(np.uint8)
    return image
