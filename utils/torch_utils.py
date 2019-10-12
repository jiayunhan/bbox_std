import torch
from torch.autograd import Variable
import numpy as np

def numpy_to_variable(image, device=torch.device('cuda:0')):
    if len(image.shape) == 3:
        x_image = np.expand_dims(image, axis=0)
    else:
        x_image = image
    x_image = Variable(torch.tensor(x_image))
    x_image = x_image.to(device)
    x_image.retain_grad()
    return x_image

def variable_to_numpy(variable):
    return variable.cpu().detach().numpy()

def convert_torch_det_output(torch_out, cs_th=0.3):
    '''convert pytorch detection model output to list of dictionary of list
        [
            {
                'scores': [0.97943294], 
                'classes': [14], 
                'boxes': [[65.1657, 17.7265, 418.3291, 314.5997]]
            }
        ]
    '''
    ret_list = []
    for temp_torch_out in torch_out:
        temp_dic = {
            'scores' : [],
            'classes' : [],
            'boxes' : []
        }
        box_list = temp_torch_out['boxes'].cpu().numpy()
        score_list = temp_torch_out['scores'].cpu().numpy()
        label_list = temp_torch_out['labels'].cpu().numpy()
        for box, score, label in zip(box_list, score_list, label_list):
            if score >= cs_th:
                temp_dic['scores'].append(score)
                temp_dic['boxes'].append(box)
                temp_dic['classes'].append(label)
        ret_list.append(temp_dic)
    return ret_list
