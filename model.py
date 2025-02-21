import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, proj_head_type='2layer', dataset='cifar10'):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        if proj_head_type == '2layer':
          self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, feature_dim, bias=True))
        elif proj_head_type == 'linear':
          self.g = nn.Sequential(nn.BatchNorm1d(2048),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2048, feature_dim, bias=True))
        elif proj_head_type == 'none':
          self.g = lambda x:x

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
