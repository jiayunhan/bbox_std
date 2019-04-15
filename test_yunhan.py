import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from torch_utils import numpy_to_variable
from torch_utils import variable_to_numpy
from torch.autograd import Variable
import numpy as np
import foolbox
from wrappers.vgg import Vgg16
from api_utils import detect_label, detect_text

with open('labels.txt','r') as inf:
    imagenet_dict = eval(inf.read())



def attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


cuda = torch.device('cuda:0')
image = cv2.imread("images/text.png")
image = cv2.resize(image, (224, 224))
image = image[...,::-1]
image = (np.transpose(image, (2, 0, 1))).astype(np.float32)
image = numpy_to_variable(image, cuda)

model = Vgg16()
k=6

'''
pred = model(image)[k]
loss = pred.std()
loss.backward(retain_graph=True)
grad = image.grad.data
x_adv = image
'''
#learning_rate = 1e-4
#optimizer = torch.optim.Adam(x_adv, lr=learning_rate)

loss = None
x_adv = image
x_adv.retain_grad()

for i in range(1000):
    preds = model(x_adv)
    internal_logits = preds[-2]
    final_logits = preds[-1]
    label = np.argmax(variable_to_numpy(final_logits), axis=1)[0]
    loss = internal_logits.std()
    
    # save image
    x_adv_np = x_adv.cpu().detach().numpy()
    x_adv_np = np.squeeze(x_adv_np)
    x_adv_np = np.transpose(x_adv_np, (1, 2, 0))
    x_adv_np = (x_adv_np * 255).astype(np.uint8)

    if (i % 50 == 0):
        Image.fromarray(x_adv_np).save('./layers_car/car_%d.jpg' % i)
        google_label = detect_text('./layers_car/car_%d.jpg' % i)
        print(i, variable_to_numpy(loss), imagenet_dict[label])
        # print(google_label)

    loss.backward(retain_graph=True)
    grad = x_adv.grad.data
    x_adv = attack(x_adv, 0.01, grad)
    x_adv.retain_grad()
