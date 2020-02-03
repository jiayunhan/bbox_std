import torchvision.models as models
import torch
from PIL import Image
from utils.image_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_gpu
import numpy as np
import torchvision
import pdb


def _save_img(imgs_np, file_path):
    img_np = np.transpose((imgs_np[0] * 255).astype(np.uint8), (1, 2, 0))
    img_pil = Image.fromarray(img_np)
    img_pil.save(file_path)
    return

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]
image_np = np.expand_dims(load_image(data_format='channels_first', fpath='./images/example.png', abs_path=True), axis=0)
image = numpy_to_variable(image_np)
_save_img(image_np, './temp_ori.png')

model = Vgg16()
internal = [i for i in range(29)]
attack = DispersionAttack_gpu(model, epsilon=16./255, step_size=1./255, steps=200)
adv = attack(image, attack_layer_idx_list=[14], internal=internal)

adv_np = variable_to_numpy(adv)
_save_img(adv_np, './temp_adv.png')

diff_np = np.abs(image_np - adv_np)
_save_img(diff_np, './temp_diff.png')

diff_amp_np = diff_np / diff_np.max()
_save_img(diff_amp_np, './temp_diff_amp_{0:.2f}.png'.format(1./diff_np.max()))

