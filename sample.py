import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack
from api_utils import detect_label_numpy
import numpy as np


# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]
image = load_image(data_format='channels_first', fname='origin.jpg')
image = numpy_to_variable(image)

model = Resnet152()
attack = DispersionAttack(model, steps=6 00)
internal = [4, 5, 6, 7]
adv = attack(image, internal=internal)
adv_np = variable_to_numpy(adv)
image = numpy_to_bytes(adv_np)
ret = detect_label_numpy(image)
print(ret)
save_image(adv_np)