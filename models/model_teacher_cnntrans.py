import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn, Tensor
import torch.utils.model_zoo as model_zoo
from Transformer import TransformerEncoder, TransformerEncoderLayer

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features
        self.distilled_features = []
        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        x = self.features(x)   # vgg network

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        #
        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x), features
    
    def regist_hook(self):
        self.distilled_features = []
        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
            self.distilled_features.append(output.detach())
        def get_transformer(model, input, output):
            self.distilled_features.append(output[0].detach())
        for name, module in self._modules['features']._modules.items():
            if name in ['4', '9', '18', '27', '36']:
                self._modules['features']._modules[name].register_forward_hook(get)
        # for i in range(len(self.encoder.layers)): #module in self._modules['encoder']._modules.items():
        #     self._modules['encoder']._modules['layers'][i].register_forward_hook(get_transformer)


def vgg19_trans(cfg = {'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(make_layers(cfg['E']))
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model