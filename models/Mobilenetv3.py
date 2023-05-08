import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(conv2d.bias, 0)
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)

class MobileNetV3Small_module(nn.Module):
    def __init__(self, direct=False):
        super(MobileNetV3Small_module, self).__init__()
        self.direct = direct
        model = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
        self.net0 = model.features[0]
        self.net1 = model.features[1]
        self.net2 = model.features[2:4]
        self.net3 = model.features[4:9]
        self.net4 = model.features[9:12]
        self.net5 = nn.Sequential(
            nn.Conv2d(96, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.net5[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.net5[0].bias, 0)
        
        if direct:
            self.transform0 = feature_transform(16, 64)
            self.transform1 = feature_transform(16, 128)
            self.transform2 = feature_transform(24, 256)
            self.transform3 = feature_transform(48, 512)
            self.transform4 = feature_transform(96, 512)

    def forward(self, x):
        x = self.net0(x)
        if not self.direct:
            feature_list = []
            feature_list.append(self.transform0(x))
        x = self.net1(x)
        if not self.direct:
            feature_list.append(self.transform1(x))
        x = self.net2(x)
        if not self.direct:
            feature_list.append(self.transform2(x))
        x = self.net3(x)
        if not self.direct:
            feature_list.append(self.transform3(x))
        x = self.net4(x)
        if not self.direct:
            feature_list.append(self.transform4(x))
        x = self.net5(x)
        if self.direct:
            return x
        else:
            return x, feature_list