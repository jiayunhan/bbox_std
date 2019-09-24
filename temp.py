from PIL import Image
import pdb
import numpy as np
from utils.image_utils import load_image

ori_img_name = '/home/yantao/workspace/datasets/imagenet5000/ori/ILSVRC2012_val_00047445.JPEG'
adv_img_name = '/home/yantao/workspace/datasets/imagenet5000/dr_vgg16_layerAt_14_eps_16_stepsize_1_steps_2000/ILSVRC2012_val_00047445.png'
ori_img = load_image(data_format='channels_first', abs_path=True, fpath=ori_img_name)
adv_img = load_image(data_format='channels_first', abs_path=True, fpath=adv_img_name)
diff = np.max(ori_img - adv_img)
pdb.set_trace()