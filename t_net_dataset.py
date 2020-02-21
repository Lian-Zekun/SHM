import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import t_im_size, fg_path, bg_path, a_path, out_path,trimap_path, training_fg_names_path, data_transform
from utils import safe_crop, crop_offset


class TNetDataset(Dataset):
    def __init__(self):
        with open('names.txt', 'r') as file:
            self.names = file.read().splitlines()

        self.transformer = data_transform

    def __getitem__(self, i):
        # 加载图片
        name = self.names[i]
        img = cv.imread(out_path + name)
        trimap = cv.imread(trimap_path + name, 0)

        # 裁剪训练对(image, trimap)为不同的大小，如 480x480, 640x640,
        # 并 resize 到 320x320，以使模型对不同尺寸更鲁棒，有助于网络更好的学习上下文和语义等高层信息.
        crop_sizes = [400, 600, 800]
        crop_size = random.choice(crop_sizes)

        x, y = crop_offset(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size, t_im_size)
        trimap = safe_crop(trimap, x, y, crop_size, t_im_size)
        
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
