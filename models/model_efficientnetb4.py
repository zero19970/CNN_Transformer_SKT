import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class LightweightCANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(LightweightCANNet, self).__init__()
        
        self.encoder = EfficientNet.from_pretrained('efficientnet-b4')
        self.backend_feat = [448, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=1792, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(448, 448, kernel_size=1, bias=False),
            nn.Conv2d(448, 448, kernel_size=1, bias=False),
            nn.Sigmoid()) for _ in range(4)])
        
        if not load_weights:
            self._initialize_weights()
    
    def forward(self, x):
        fv = self.encoder.extract_features(x)
        
        attention_outputs = []
        for attention_layer in self.attention_layers:
            pooled = nn.functional.adaptive_avg_pool2d(fv, attention_layer[0].out_channels)
            attention_output = attention_layer(pooled)
            upsampled = nn.functional.interpolate(attention_output, size=(fv.shape[2], fv.shape[3]), mode='bilinear', align_corners=True)
            attention_outputs.append(upsampled)
        
        fi = sum(attention_outputs) / len(attention_outputs)
        x = torch.cat((fv, fi), 1)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
