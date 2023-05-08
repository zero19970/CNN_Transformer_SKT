from Mobilenetv3 import MobileNetV3Small_module
import torch
from torch import nn
import torch.nn.functional as F
from Transformer import TransformerEncoder, TransformerEncoderLayer

class VGG_Trans_student(nn.Module):
    def __init__(self, features, trans_layer):
        super(VGG_Trans_student, self).__init__()
        self.features = features
        self.distilled_features = []
        d_model = 512
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=2048,
                                                dropout=0.1, activation="relu", normalize_before=False)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, trans_layer, if_norm)
        # self.reg_layer_0 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 1, 1)
        # )
        self.reg_layer_0 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        x, cnn_features = self.features(x)   # vgg network
        self.distilled_features = cnn_features

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, trans_features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)

        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x)

def vgg19_trans_student(trans_layer=1, direct=True):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    
    model = VGG_Trans_student(MobileNetV3Small_module(direct=direct), trans_layer)
    return model