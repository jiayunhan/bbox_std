import copy
import numpy as np
from torch.autograd import Variable
import torch

import pdb

class DispersionAttack(object):
    def __init__(self, model, epsilon=0.01, steps=100):

        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)

    def __call__(self, X_nat, internal=[]):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat_np = X_nat.numpy()
        for p in self.model.parameters():
            p.requires_grad = False
        
        X = np.copy(X_nat_np)
        
        for i in range(self.steps):
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
            internal_logits, pred = self.model.prediction(X_var, internal=internal)
            logit = internal_logits[-1]
            loss = logit.std()
            if(i % 20 == 0):
                print(loss)
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()

            X -= self.epsilon * np.sign(grad)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return torch.from_numpy(X)