import torchvision.models as models
import torch

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()

    def forward(self, x):
        results = []
        last_layer = self.model(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if isinstance(model, torch.nn.modules.conv.Conv2d):
                results.append(x)
        results.append(last_layer)
        return results

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.model = models.vgg19(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()

    def forward(self, x):
        results = []
        last_layer = self.model(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if isinstance(model, torch.nn.modules.conv.Conv2d):
                results.append(x)
        results.append(last_layer)
        return results
