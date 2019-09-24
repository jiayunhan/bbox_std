import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from torch_utils import numpy_to_variable
from torch.autograd import Variable
import numpy as np
import foolbox
from wrappers.vgg import Vgg16

# origin_model = models.vgg16_bn(pretrained=True).cuda().eval()
# modules = list(origin_model.features)
# modules = list(origin_model.modules())[1:]



def attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


cuda = torch.device('cuda:0')

# x_image = Variable(torch.zeros(1, 3, 244, 244), requires_grad=True)
'''
image, label = foolbox.utils.imagenet_example(data_format='channels_first')
origin = np.transpose(image, (1, 2, 0)).astype(np.uint8)
Image.fromarray(origin).save('/home/yunhan/Documents/bh-asia/images/0000.png')
'''

image = cv2.imread("images/origin.jpg")
image = cv2.resize(image, (224, 224))
image = image[...,::-1]
image = (np.transpose(image, (2, 0, 1))).astype(np.float32)

x_image = numpy_to_variable(image, cuda)

model = Vgg16()

k = 8


pred = model(x_image)[k]
loss = pred.std()
#loss = torch.norm(pred, 2)
loss.backward(retain_graph=True)
grad = x_image.grad.data
print(k, pred.shape, loss)

x_pert = x_image


for i in range(8, 8):
    x_pert = attack(x_pert, .01, grad)
    x_pert.retain_grad()
    preds = model(x_pert)
    pred = preds[k]
    print(pred.shape)
    # loss = torch.norm(pred, 2)
    loss = pred.std()
    loss.backward(retain_graph=True)
    grad = x_pert.grad.data
    print(loss)

diff_inf = torch.dist(x_pert, x_image, p=float("inf"))
diff_2 = torch.dist(x_pert, x_image, p=2)
print(diff_inf, diff_2) 

x_np = x_pert.cpu().detach().numpy()
x_np = np.squeeze(x_np)
x_np = np.transpose(x_np, (1, 2, 0))
x_np = (x_np * 255).astype(np.uint8)
Image.fromarray(x_np).save('./layers_car/large_%d.jpg' % k)
Image.fromarray(x_np).show()

#indice = torch.argmax(origin_model(x_pert))
#print(indice)