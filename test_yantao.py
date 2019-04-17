import torchvision
from PIL import Image
import torch
import numpy as np

from attacks.mifgsm import MomentumIteratorAttack
from attacks.linf_pgd import LinfPGDAttack
import pdb

image = Image.open("images/origin.jpg").convert('RGB')
image = image.resize((224, 224))

model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=0, epsilon=16/255, steps=8, step_size=4/255, random_start=False)
#attack = LinfPGDAttack(model, epsilon=16/255, k=8, a=4/255, random_start=False)

image_np_nchw = np.expand_dims((np.transpose(np.array(image), (2, 0, 1))).astype(np.float32), axis=0) / 255
image_torch_nchw = torch.from_numpy(image_np_nchw).float()

pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
print('Pred. nat. : ', np.argmax(pred_nat))

label = np.argmax(pred_nat)
label_tensor = torch.tensor(np.array([label]))
image_adv = attack(image_torch_nchw, label_tensor)
pred_adv = model(image_adv.cuda()).detach().cpu().numpy()
print('Pred. adv. : ', np.argmax(pred_adv))

image_ori_pil = Image.fromarray((np.transpose(image_torch_nchw[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
image_ori_pil.save('test_ori_out.png')
image_adv_pil = Image.fromarray((np.transpose(image_adv[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
image_adv_pil.save('test_adv_out.png')

