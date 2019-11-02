import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision
import torch

from utils.image_utils import load_image, draw_masks

import pdb  

output_dir = '/home/yantao/compare_seg'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)


dir_path = '/home/yantao/workspace/datasets/baseline_COCO'
dir_advs = {
    'dr' : '/home/yantao/workspace/datasets/COCO2017_1000/dr_vgg16_layerAt_14_eps_16_stepsize_4.0_steps_100_lossmtd_std',
    'pgd' : 'pgd_inception_v3_layerAt_0_eps_16_stepsize_1.0_steps_20_lossmtd_',
    'mifgsm' : 'mifgsm_inception_v3_layerAt_0_eps_16_stepsize_1.0_steps_20_lossmtd_',
    'dim' : 'dim_inception_v3_layerAt_0_eps_16_stepsize_1.0_steps_20_lossmtd_',
    'ti' : 'tidim_inception_v3_layerAt_0_eps_16_stepsize_1.6_steps_20_lossmtd_'
}

num_classes = 21
model = torchvision.models.segmentation.deeplabv3_resnet101(
    pretrained=True, 
    progress=True, 
    num_classes=21
)
model = model.cuda().eval()
img_size = (520, 520)
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size), 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean=img_mean, std=img_std)])

dir_ori = os.path.join(dir_path, 'ori')
img_names = os.listdir(dir_ori)
for idx, img_name in enumerate(img_names):
    img_names[idx] = os.path.splitext(img_name)[0]

for img_name in tqdm(img_names):
    img_paths = {}
    img_paths['ori'] = os.path.join(dir_path, 'ori', img_name + '.jpg')
    for key, val in dir_advs.items():
        if key == 'dr':
            img_paths[key] = os.path.join(val, img_name + '.png')
        else:
            img_paths[key] = os.path.join(dir_path, val, img_name + '.png')
    for perfix, img_path in img_paths.items():
        image_np = load_image(data_format='channels_last', shape=img_size, bounds=(0, 255), abs_path=True, fpath=img_path)
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        image_pil.save(os.path.join(output_dir, img_name + '.png'))

        image_var = img_transforms(Image.open(img_path).convert('RGB')).unsqueeze_(axis=0).cuda()
        with torch.no_grad():
            seg_out = model(image_var)['out']
        mask = np.argmax(seg_out.data.cpu().numpy(), axis=1)[0]
        draw_masks(
            os.path.join(output_dir, img_name + '.png'), 
            mask, 
            num_classes, 
            from_path=True, 
            out_file=os.path.join(output_dir, img_name + '_' + perfix + '.png')
        )

    image_np = load_image(data_format='channels_last', shape=img_size, bounds=(0, 255), abs_path=True, fpath=img_paths['ori'])
    image_pil = Image.fromarray(image_np.astype(np.uint8))
    image_pil.save(os.path.join(output_dir, img_name + '.png'))