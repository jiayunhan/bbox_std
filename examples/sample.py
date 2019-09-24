import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from api_utils import detect_label_numpy
import numpy as np
import torchvision
import pdb

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]
image_np = load_image(data_format='channels_first', fname='origin.jpg')
image = numpy_to_variable(image_np)

'''
model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=0.5, epsilon=60./255, steps=1000, step_size=1./255, random_start=False)
image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
label = np.argmax(pred_nat)
label_tensor = torch.tensor(np.array([label]))
adv = attack(image_torch_nchw, label_tensor)
'''


model = Vgg16()
internal = [i for i in range(29)]
attack = DispersionAttack(model, epsilon=60./255, step_size=1./255, steps=1000)
adv = attack(image, internal=internal)



adv_np = variable_to_numpy(adv)
image = numpy_to_bytes(adv_np)
ret = detect_label_numpy(image)
print(ret)
save_image(adv_np)

linf = int(np.max(abs(image_np - adv_np)) * 255)
print('linf: ', linf)