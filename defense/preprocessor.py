import torchvision.models as models
import torch
import numpy as np

import pdb


class DIM_tranform(torch.nn.Module):
    def __init__(self, image_h, image_w, image_resize, resize_back=True):
        super(DIM_tranform, self).__init__()
        self.shape = [image_h, image_w]
        self.image_resize = image_resize
        self.resize_back = resize_back

    def __call__(self, input_tensor):
        assert self.shape[0] < self.image_resize and self.shape[1] < self.image_resize
        rnd = np.random.randint(self.shape[1], self.image_resize)
        input_upsample = torch.nn.functional.interpolate(input_tensor, size=(rnd, rnd), mode='nearest')
        h_rem = self.image_resize - rnd
        w_rem = self.image_resize - rnd
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padder = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.0)
        input_padded = padder(input_upsample)
        if self.resize_back:
            input_padded_resize = torch.nn.functional.interpolate(input_padded, size=self.shape, mode='nearest')
            return input_padded_resize
        else:
            return input_padded