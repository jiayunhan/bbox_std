import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from api_utils import detect_label_numpy
import numpy as np
import torchvision
from api_utils import detect_label_file
from tqdm import tqdm
import os
import pdb

    
with open('labels.txt','r') as inf:
    imagenet_dict = eval(inf.read())

model = Vgg16()
internal = [i for i in range(29)]

attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000)

image_np = load_image(data_format='channels_first', fname='cat.jpg')
image_var = numpy_to_variable(image_np)

internal_logits_var, pred_nat_var = model.prediction(image_var, internal=internal)

pred_nat = pred_nat_var.detach().cpu().numpy()
gt_label = np.argmax(pred_nat)
pred_cls = imagenet_dict[gt_label]
print(pred_cls)

for layer_idx, intermediate_logit_var in internal_logits_var:
    pdb.set_trace()



adv, info_dict = attack(image_var, 
                        attack_layer_idx=3,
                        internal=internal)

adv_np = variable_to_numpy(adv)
linf = int(np.max(abs(image_np - adv_np)) * 255)
print('linf: ', linf)
l1 = np.mean(abs(image_np - adv_np)) * 255
print('l1: ', l1)
l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
print('l2: ', l2)
print(" ")


'''
model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=1.0, epsilon=16./255, steps=2000, step_size=1./255, random_start=False)

total_samples = 100
success_attacks = 0
for idx, temp_image_name in enumerate(images_name):
    print('idx: ', idx)

    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)

    Image.fromarray(np.transpose((image_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label = detect_label_file('./out/ori.jpg')
    if len(google_label) > 0:
        pred_cls = google_label[0].description
    else:
        pred_cls = None
    print(pred_cls)

    image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
    pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
    label = np.argmax(pred_nat)
    label_tensor = torch.tensor(np.array([label]))
    adv = attack(image_torch_nchw, label_tensor)
    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp.jpg')
    google_label = detect_label_file('./out/temp.jpg')
    if len(google_label) > 0:
        output_cls = google_label[0].description
    else:
        output_cls = None
    print(output_cls)
    print(" ")

    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    print(" ")

    if output_cls != pred_cls:
        success_attacks += 1

print('attack success rate: ', float(success_attacks) / float(total_samples))
'''