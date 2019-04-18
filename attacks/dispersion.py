import copy
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F

import pdb

class DispersionAttack(object):
    def __init__(self, model, epsilon=0.063, step_size=0.004, steps=100, regularization_weight=1):
        
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)
        self.regularization_weight = regularization_weight

    def __call__(self, X_nat, internal=[]):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat_np = X_nat.cpu().numpy()
        for p in self.model.parameters():
            p.requires_grad = False
        
        X = np.copy(X_nat_np)
        
        for i in range(self.steps):
            X_nat_var = Variable(torch.from_numpy(X_nat_np).cuda(), requires_grad=False, volatile=False)
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
            internal_logits, pred = self.model.prediction(X_var, internal=internal)
            logit = internal_logits[14]
            loss = logit.std() #+ self.regularization_weight * F.l1_loss(X_nat_var, X_var, reduction='mean')
            if(i % 20 == 0):
                print(loss)
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()

            X -= self.step_size * np.sign(grad)
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return torch.from_numpy(X)