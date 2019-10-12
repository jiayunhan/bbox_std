import torchvision.models as models
import torch
import pdb

class Inception_v3(torch.nn.Module):
    '''
        A : 5, 6, 7
        B : 8
        C : 9, 10, 11, 12
    '''
    def __init__(self):
        super(Inception_v3, self).__init__()
        self.model = models.inception_v3(pretrained=True).cuda().eval()
        features = list(self.model.children())
        #print(len(features))
        #for ii, model in enumerate(features):
        #    print(ii, model)
        self.features = torch.nn.ModuleList(features)

    def prediction(self, x, internal=[]):
        pred = self.model(x)
        hit_cnt = 0
        if len(internal) == 0:
            return pred
        
        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if(ii in internal):
                hit_cnt += 1
                layers.append(x)
            if(hit_cnt==len(internal)):
                break
        return layers, pred

if __name__ == "__main__":
    Inception_v3()