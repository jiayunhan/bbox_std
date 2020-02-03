import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes, visualize_features, visualize_features_compare
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

attack = DispersionAttack_opt(model, epsilon=16./255, steps=200)

image_np = load_image(data_format='channels_first', fname='example.png')

image_pil = Image.fromarray(np.transpose((image_np * 255).astype(np.uint8), (1, 2, 0)))
image_pil.save(os.path.join("results", 'original.png'))

image_var = numpy_to_variable(image_np)

internal_logits_var, pred_nat_var = model.prediction(image_var, internal=internal)

pred_nat = pred_nat_var.detach().cpu().numpy()
gt_label = np.argmax(pred_nat)
print(gt_label)
pred_cls = imagenet_dict[gt_label]
print(pred_cls)

'''
for layer_idx, intermediate_logit_var in enumerate(tqdm(internal_logits_var)):
    intermediate_features = intermediate_logit_var[0].detach().cpu().numpy()
    visualize_features(intermediate_features, output_dir='results', file_prefix='ori_{0:02d}_'.format(layer_idx), data_format='channels_first', only_first_channel=True)
'''

# ----------------------------------------------------------------
adv, info_dict = attack(image_var, 
                        attack_layer_idx=10,
                        internal=internal)

internal_logits_adv_var, pred_adv_var = model.prediction(adv.cuda(), internal=internal)
pred_adv = pred_adv_var.detach().cpu().numpy()
pd_label = np.argmax(pred_adv)
print(pd_label)
pred_cls = imagenet_dict[pd_label]
print(pred_cls)

'''
for layer_idx, intermediate_logit_adv_var in enumerate(tqdm(internal_logits_adv_var)):
    intermediate_features = intermediate_logit_adv_var[0].detach().cpu().numpy()
    visualize_features(intermediate_features, output_dir='results', file_prefix='adv_{0:02d}_'.format(layer_idx), data_format='channels_first', only_first_channel=True)
'''

for layer_idx, (intermediate_logit_var, intermediate_logit_adv_var) in enumerate(tqdm(zip(internal_logits_var, internal_logits_adv_var))):
    intermediate_features = intermediate_logit_var[0].detach().cpu().numpy()
    intermediate_adv_features = intermediate_logit_adv_var[0].detach().cpu().numpy()
    visualize_features_compare(intermediate_features, intermediate_adv_features, output_dir='results', file_prefix='compare_{0:02d}_'.format(layer_idx), data_format='channels_first', only_first_channel=True)

adv_np = variable_to_numpy(adv)

image_pil = Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0)))
image_pil.save(os.path.join("results", 'adv_at_{0:02d}.png'.format(10)))

linf = int(np.max(abs(image_np - adv_np)) * 255)
print('linf: ', linf)
l1 = np.mean(abs(image_np - adv_np)) * 255
print('l1: ', l1)
l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
print('l2: ', l2)
print(" ")


