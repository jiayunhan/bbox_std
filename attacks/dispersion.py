import copy
import os
import shutil
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.torch_utils import numpy_to_variable, variable_to_numpy
from utils.api_utils import detect_label_file, detect_objects_file, googleDet_to_Dictionary
from utils.image_utils import save_bbox_img

import pdb


class DispersionAttack_gpu(object):
    def __init__(self, model, epsilon=16/255., step_size=0.004, steps=10, loss_mtd='std'):
        
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model).cuda()
        self.loss_mtd = loss_mtd
        if loss_mtd == 'std':
            self.loss_fn = self._std_loss
        elif loss_mtd == 'l1smooth_zero_loss':
            self.loss_fn = self._l1smooth_zero_loss
        elif loss_mtd == 'l1smooth_avg_loss':
            self.loss_fn = self._l1smooth_avg_loss
        elif loss_mtd == 'selective_loss':
            self.loss_fn = self._selective_loss
            self.selected_idx_list = None
        else:
            raise ValueError('')

    def __call__(self, X_nat_var, attack_layer_idx_list, internal):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        X_var = copy.deepcopy(X_nat_var)
        for i in range(self.steps):
            X_var = X_var.requires_grad_()
            internal_logits, _ = self.model.prediction(X_var, internal=internal)
            logit_list = [internal_logits[x] for x in attack_layer_idx_list]

            if self.loss_mtd == 'selective_loss':
                if self.selected_idx_list is None:
                    self.selected_idx_list = self.__init_selective_loss(logit_list, 0.5)

            #print(i, ' , std: ', [x.std() for x in logit_list])

            loss_list = [self.loss_fn(logit) for self.logit_idx, logit in enumerate(logit_list)]
            loss = None
            for temp_loss in loss_list:
                if loss is None:
                    loss = temp_loss
                else:
                    loss = loss + temp_loss 
            loss = loss.mean()
            #print(i, ' , ', loss)

            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data

            X_var = X_var.detach() + self.step_size * grad.sign_()
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)
        return X_var.detach()

    def _std_loss(self, logit):
        return -1 * logit.view(logit.shape[0], -1).std(1)

    def _l1smooth_zero_loss(self, logit):
        l1_smooth_loss = torch.nn.SmoothL1Loss().cuda()
        gt = torch.zeros(logit.shape, requires_grad=False, dtype=torch.float32).cuda()
        return -1 * l1_smooth_loss(logit, gt)

    def _l1smooth_avg_loss(self, logit):
        l1_smooth_loss = torch.nn.SmoothL1Loss().cuda()
        logit = logit.view(logit.shape[0], -1)
        avg = logit.mean(1).unsqueeze_(-1)
        gt = avg.repeat((1, logit.shape[-1])).detach()
        return -1 * l1_smooth_loss(logit, gt)

    def _selective_loss(self, logit):
        l1_smooth_loss = torch.nn.SmoothL1Loss().cuda()
        logit = logit.view(logit.shape[0], -1)
        selected_id = self.selected_idx_list[self.logit_idx]
        loss = None
        for batch_idx, curt_logit in enumerate(logit):
            curt_selected_id = selected_id[batch_idx]
            selected_curt_logit = curt_logit[curt_selected_id]
            gt = torch.zeros(selected_curt_logit.shape, requires_grad=False, dtype=torch.float32).cuda()
            curt_loss = l1_smooth_loss(selected_curt_logit, gt)
            if loss is None:
                loss = curt_loss
            else:
                loss += curt_loss
        return -1 * loss / logit.shape[0]

    def __init_selective_loss(self, logit_list, th_ratio):
        ret_list = []
        for _, logit in enumerate(logit_list):
            logit = logit.view(logit.shape[0], -1)
            logit_cp = logit.clone().detach().cpu().numpy()
            thresholds = logit_cp.max(axis=-1)
            arg_idx_list = []
            for th_idx, th_max in enumerate(thresholds):
                th = th_max * th_ratio
                arg_id = np.argwhere(logit_cp[th_idx] >= th).reshape(-1)
                arg_idx_list.append(arg_id)
            ret_list.append(arg_idx_list)

            #top_number = int(logit.shape[-1] * ratio)
            #_, large_idx = torch.topk(logit, top_number, dim=-1)
            #large_idx.detach()
            #ret_list.append(large_idx)

        return ret_list


class DispersionAttack_opt(object):
    def __init__(self, model, epsilon=0.063, learning_rate=5e-2, steps=100, regularization_weight=0, is_test_api=False, is_test_model=False):
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)
        self.regularization_weight = regularization_weight
        self.is_test_api = is_test_api
        self.is_test_model = is_test_model
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        assert (self.is_test_api and self.is_test_model) == False, "At most one of the test can be activated."

    def __call__(self, X_nat, attack_layer_idx=-1, internal=[], test_steps=None, gt_label=None, test_model=None):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.is_test_model:
            assert test_model is not None, "test_model has to be specified when is_test_model is activated."

        info_dict = {}

        X_nat_np = X_nat.cpu().numpy()
        for p in self.model.parameters():
            p.requires_grad = False
        
        X = np.copy(X_nat_np)
        optimizer = AdamOptimizer(X.shape)

        ori_label = None
        for i in range(self.steps):
            X_nat_var = Variable(torch.from_numpy(X_nat_np).cuda(), requires_grad=False, volatile=False)
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)

            internal_logits, pred = self.model.prediction(X_var, internal=internal)

            if i == 0:
                ori_label = torch.max(pred[0], 0)[1]
                ori_label = ori_label.unsqueeze(0)
            cls_loss = self.loss_fn(pred, ori_label)
            logit = internal_logits[attack_layer_idx]
            loss = -1 * logit.std() + 0. * cls_loss + self.regularization_weight * F.l1_loss(X_nat_var, X_var, reduction='mean')
            loss.backward()

            grad = X_var.grad.data.cpu().numpy()
            X += optimizer(grad, learning_rate=self.learning_rate)

            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

            if self.is_test_model and i % test_steps == 0:
            
                adv_np = X
                adv_var = torch.from_numpy(adv_np).cuda()
                pred = test_model(adv_var).detach().cpu().numpy()
                pred_label = np.argmax(pred)
                if gt_label is not None:
                    if gt_label != pred_label:
                        info_dict['end_epoch'] = i
                        info_dict['det_label'] = pred_label
                        info_dict['loss'] = loss.detach().cpu().numpy()
                        return torch.from_numpy(X), info_dict

            if self.is_test_api and i % test_steps == 0:
                adv_np = X
                Image.fromarray(np.transpose((adv_np[0] * 255.).astype(np.uint8), (1, 2, 0))).save('./temp/temp_dispersion_opt.jpg')
                google_label = detect_label_file('./temp/temp_dispersion_opt.jpg')
                if len(google_label) > 0:
                    pred_cls = google_label[0].description
                else:
                    pred_cls = 'none'

                if gt_label is not None:
                    if gt_label != pred_cls and gt_label != 'none':
                        info_dict['end_epoch'] = i
                        info_dict['det_label'] = pred_cls
                        info_dict['loss'] = loss.detach().cpu().numpy()
                        return torch.from_numpy(X), info_dict

        return torch.from_numpy(X), info_dict


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.

    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized.

    """

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate,
                 beta1=0.98, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns the
        change that should be applied to the variable.

        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loww w.r.t. to the variable.
        learning_rate: float
            the learning rate in the current iteration.
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients.
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients.
        epsilon: float
            small value to avoid division by zero.

        """
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)