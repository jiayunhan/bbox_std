import torch
import torchvision
import numpy as np
import os
from PIL import Image

from image_utils import load_image, save_image, save_bbox_img, numpy_to_bytes
from torch_utils import numpy_to_variable, variable_to_numpy
from api_utils import detect_objects_file, googleDet_to_Dictionary
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from mAP import save_detection_to_file

import pdb


dataset_dir = "/home/yantao/datasets/imagenet_100image/"
images_name = os.listdir(dataset_dir)

model = Vgg16()
internal = [i for i in range(29)]
#attack = DispersionAttack(model, epsilon=16./255, step_size=1./255, steps=2000, is_test_api=True)
attack = DispersionAttack_opt(model, epsilon=64./255, steps=2000)

total_samples = 100
success_attacks = 0
for idx, temp_image_name in enumerate(images_name):
    print('idx: ', idx)
    temp_image_name_noext = os.path.splitext(temp_image_name)[0]
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)
    adv = image

    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label = detect_objects_file('./out/ori.jpg')
    google_label = googleDet_to_Dictionary(google_label, adv_np.shape[-2:])
    save_detection_to_file(google_label, os.path.join('out', 'DispersionAttack_opt_det_out', 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
    print(google_label)
    if google_label:
        save_bbox_img('./out/ori.jpg', google_label['boxes'], out_file='temp_ori_box.jpg')
    else:
        save_bbox_img('./out/ori.jpg', [], out_file='temp_ori_box.jpg')

    temp_attack_success = 0
    adv, _ = attack(image, 
                    attack_layer_idx=14, 
                    internal=internal, 
                    test_steps=200
                    )
    adv_np = variable_to_numpy(adv)

    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp_adv.jpg')
    google_label = detect_objects_file('./out/temp_adv.jpg')
    google_label = googleDet_to_Dictionary(google_label, adv_np.shape[-2:])
    save_detection_to_file(google_label, os.path.join('out', 'DispersionAttack_opt_det_out', 'pd', temp_image_name_noext + '.txt'), 'detection')
    print(google_label)
    if google_label:
        save_bbox_img('./out/temp_adv.jpg', google_label['boxes'], out_file='temp_adv_box.jpg')
    else:
        save_bbox_img('./out/temp_adv.jpg', [], out_file='temp_adv_box.jpg')

    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    pdb.set_trace()


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