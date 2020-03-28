import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import *


# 数据标准化
data_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


with open(training_fg_names_path) as f:
    fg_files = f.read().splitlines()   
with open(training_bg_names_path) as f:
    bg_files = f.read().splitlines()
with open(data_names_path) as file:
    names = file.read().splitlines()


def crop_offset(trimap, crop_size):
    """以未知区域的像素为中心，随机裁剪 320x320 的 (image, trimap) 对，以增加采样空间."""
    y_indices, x_indices = np.where(trimap == 128)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        index = np.random.randint(low=0, high=num_unknowns)
        center_x = x_indices[index]
        center_y = y_indices[index]
        x = max(0, center_x - crop_size // 2)
        y = max(0, center_y - crop_size // 2)
    return x, y


# 应该分为原图片大小大于目标大小(crop_size)需要进行裁剪
# 原图大小小于输出图片大小(im_size) 或 目标大小(crop_size)大于输出图片大小(im_size)直接resize
def safe_crop(mat, x, y, crop_size, im_size, type=0):
    if len(mat.shape) == 2:
        ret = np.zeros((crop_size, crop_size), np.uint8)
    else:
        ret = np.zeros((crop_size, crop_size, 3), np.uint8)
    crop = mat[y:y + crop_size, x:x + crop_size]
    h, w = crop.shape[:2] # 若原图片小于目标大小,则crop shape不会保证预期的crop_size,需要重新得到shape
    ret[0:h, 0:w] = crop
    if crop_size != im_size:
        if type == 0:
            ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_AREA)
        else:  # 对应trimap图像使用cv.INTER_NEAREST
            ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret


class TnetDataset(Dataset):
    def __init__(self):
        self.names = names
        self.transformer = data_transform

    def __getitem__(self, i):
        # 加载图片
        name = self.names[i]
        img = cv.imread(out_path + name)
        trimap = cv.imread(trimap_path + name, 0)

        # 裁剪训练对(image, trimap)为不同的大小，如 480x480, 640x640,
        # 并 resize 到 320x320，以使模型对不同尺寸更鲁棒，有助于网络更好的学习上下文和语义等高层信息.
        crop_sizes = [320, 480, 640]
        crop_size = random.choice(crop_sizes)

        x, y = crop_offset(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size, im_size)
        trimap = safe_crop(trimap, x, y, crop_size, im_size, 1)
        
        # trimap should be 3 classes : fg, bg. unsure
        trimap[trimap==0] = 0
        trimap[trimap==128] = 1
        trimap[trimap==255] = 2 

        # 对每个训练对(image, trimap) 随机镜像处理.
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        
        trimap = np.ascontiguousarray(trimap)
        trimap = torch.from_numpy(trimap.astype(np.float32))

        return img, trimap

    def __len__(self):
        return len(self.names)
        
        
class SHMDataset(Dataset):
    def __init__(self):
        self.names = names
        self.transformer = data_transform

    def __getitem__(self, i):
        # 加载图片
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        fg_name = fg_files[fcount]
        
        img = cv.imread(out_path + name)
        trimap = cv.imread(trimap_path + name, 0)
        alpha = cv.imread(a_path + fg_name, 0)

        # 裁剪训练对(image, trimap)为不同的大小，如 480x480, 640x640,
        # 并 resize 到 320x320，以使模型对不同尺寸更鲁棒，有助于网络更好的学习上下文和语义等高层信息.
        crop_sizes = [320, 480, 640]
        crop_size = random.choice(crop_sizes)

        x, y = crop_offset(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size, im_size)
        alpha = safe_crop(alpha, x, y, crop_size, im_size)
        trimap = safe_crop(trimap, x, y, crop_size, im_size, 1)
        
        # trimap should be 3 classes : fg, bg. unsure
        mask = np.zeros(trimap.shape)
        mask[trimap==0] = 0
        mask[trimap==128] = 1
        mask[trimap==255] = 2 

        # 对每个训练对(image, trimap) 随机镜像处理.
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)
            mask = np.fliplr(mask)

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # RGB
        transform_img = transforms.ToPILImage()(img)
        transform_img = self.transformer(transform_img)
        mask = torch.from_numpy(mask.astype(np.float32))
        alpha = torch.from_numpy(alpha.astype(np.float32) / 255.)
        alpha.unsqueeze_(dim=0)
        # trimap = np.ascontiguousarray(trimap)  # 将一个内存不连续存储的数组转换为内存连续存储的数组
        trimap = torch.from_numpy(trimap.astype(np.float32))
        trimap.unsqueeze_(dim=0)

        return transform_img, img, alpha, trimap, mask

    def __len__(self):
        return len(self.names)
