# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F

from config import channels

VGG16_BN_MODEL_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'


def encoder_layers(channels, in_channels=3):
    layers = []
    for v in channels:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def decoder_layers(channels):
    layers = []
    layers += [nn.Conv2d(channels[0], channels[0], kernel_size=5, padding=2),
               nn.BatchNorm2d(channels[0]),
               nn.ReLU(inplace=True)]
    for i in range(len(channels) - 1):
        layers += [nn.MaxUnpool2d(kernel_size=2, stride=2),
                   nn.Conv2d(channels[i], channels[i+1], kernel_size=5, padding=2),
                   nn.BatchNorm2d(channels[i+1]),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    
    def __init__(self, encoder_channels):
        super(VGG, self).__init__()
        self.features = encoder_layers(encoder_channels, in_channels=4)
        self._indices = None
        self._unpool_shapes = None
        
    def forward(self, x):
        self._indices = []
        self._unpool_shapes = []
        for layer in self.features:
            if isinstance(layer, nn.modules.pooling.MaxPool2d):
                self._unpool_shapes.append(x.size())
                x, indices = layer(x)
                self._indices.append(indices)
            else:
                x = layer(x)
        return x
    
    
def init_vgg16_bn(channels, pretrained=True, progress=True):
    model = VGG(channels)
    if pretrained:
        state_dict = tv.models.utils.load_state_dict_from_url(
            VGG16_BN_MODEL_URL, progress=progress)
        conv1_weight_name = 'features.0.weight'
        conv1_weight = model.state_dict()[conv1_weight_name]
        conv1_weight[:, :3, :, :] = state_dict[conv1_weight_name]
        conv1_weight[:, 3, :, :] = torch.tensor(0)
        state_dict[conv1_weight_name] = conv1_weight
        model.load_state_dict(state_dict, strict=False)
    return model


class DIM(nn.Module):
    """Deep Image Matting."""
    
    def __init__(self, encoder_layers, decoder_channels):
        super(DIM, self).__init__()
        # Encoder
        self.encoder_layers = encoder_layers
        # Decoder
        self.decoder_layers = decoder_layers(decoder_channels)
        # Prediction
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder_layers(x)
        indices = self.encoder_layers._indices[::-1]
        unpool_shapes = self.encoder_layers._unpool_shapes[::-1]
        index = 0
        for layer in self.decoder_layers:
            if isinstance(layer, torch.nn.modules.pooling.MaxUnpool2d):
                x = layer(x, indices[index], output_size=unpool_shapes[index])
                index += 1
            else:
                x = layer(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x
        
        
def get_m_net_model():
    vgg_encoder_channels = channels.get('vgg').get('encoder')
    vgg_decoder_channels = channels.get('vgg').get('decoder')
    vgg = init_vgg16_bn(vgg_encoder_channels)
    dim = DIM(vgg, vgg_decoder_channels)
    return dim
