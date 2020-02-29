# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import segmentation_models_pytorch as smp

VGG16_BN_MODEL_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'


channels = {
    'vgg': {
        'encoder': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                    'M', 512, 512, 512],
        'decoder': [512, 256, 128, 64, 64]
    },
}


def encoder_layers(channels, in_channel=3):
    layers = []
    for v in channels:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channel = v
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
    
    def __init__(self, encoder_channels, in_channel):
        super(VGG, self).__init__()
        self.features = encoder_layers(encoder_channels, in_channel=in_channel)
        self._indices = None  # 最大值位置索引
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


class Mnet(nn.Module):
    
    def __init__(self, encoder_layers, decoder_channels):
        super(Mnet, self).__init__()
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
        
        
class SHM(nn.Module):
    """ e2e """
    def __init__(self, t_net, m_net):
        super(SHM, self).__init__()
        self.t_net = t_net
        self.m_net = m_net
        
    def forward(self, input):
    	# trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)
        
        # 二次裁剪
        # crop_sizes = [320, 480, 640]
        # crop_size = random.choice(crop_sizes)
        # x = random.randint(0, 800-crop_sizes)
        # y = random.randint(0, 800-crop_sizes)
        
        # trimap_softmax = safe_crop(trimap_softmax, x, y, crop_size, m_im_size)
        # input = safe_crop(input, x, y, crop_size, m_im_size)   

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
        
        
def init_vgg16_bn(channels, in_channel, pretrained=True, progress=True):
    model = VGG(channels, in_channel)
    if pretrained:
        state_dict = tv.models.utils.load_state_dict_from_url(
            VGG16_BN_MODEL_URL, progress=progress)
        conv1_weight_name = 'features.0.weight'
        conv1_weight = model.state_dict()[conv1_weight_name]
        conv1_weight[:, :3, :, :] = state_dict[conv1_weight_name]
        for i in range(3, in_channel):
            conv1_weight[:, i, :, :] = torch.tensor(0)
        state_dict[conv1_weight_name] = conv1_weight
        model.load_state_dict(state_dict, strict=False)
    return model
    
    
def get_t_net_model():
    model = smp.PSPNet('resnet50', classes=3)
    return model 
        
        
def get_m_net_model(in_channel):
    vgg_encoder_channels = channels.get('vgg').get('encoder')
    vgg_decoder_channels = channels.get('vgg').get('decoder')
    vgg = init_vgg16_bn(vgg_encoder_channels, in_channel)
    dim = Mnet(vgg, vgg_decoder_channels)
    return dim


def get_shm_model():
    t_net = get_t_net_model()
    m_net = get_m_net_model(6)
    shm = SHM(t_net, m_net)
       
