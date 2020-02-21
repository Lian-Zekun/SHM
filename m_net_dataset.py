import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import m_im_size, fg_path, bg_path, a_path, trimap_path, out_path, training_fg_names_path, training_bg_names_path, data_transform
from utils import safe_crop, crop_offset

with open(training_fg_names_path) as f:
    fg_files = f.read().splitlines()
    
with open(training_bg_names_path) as f:
    bg_files = f.read().splitlines()


class DIMDataset(Dataset):
    def __init__(self):
        filename = 'names.txt'
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()

        self.transform = data_transform

    def __getitem__(self, i):
        # 加载图片
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        fg_name = fg_files[fcount]
        bg_name = bg_files[bcount]
        
        img = cv.imread(out_path + name)
        fg = cv.imread(fg_path + fg_name)
        bg = cv.imread(bg_path + bg_name)
        trimap = cv.imread(trimap_path + name, 0)
        alpha = cv.imread(a_path + fg_name, 0)

        # 裁剪训练对(image, trimap)为不同的大小，如 480x480, 640x640,
        # 并 resize 到 320x320，以使模型对不同尺寸更鲁棒，有助于网络更好的学习上下文和语义等高层信息.
        crop_sizes = [320, 480, 640]
        crop_size = random.choice(crop_sizes)

        x, y = crop_offset(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size, m_im_size)
        fg = safe_crop(fg, x, y, crop_size, m_im_size)
        bg = safe_crop(bg, x, y, crop_size, m_im_size)
        alpha = safe_crop(alpha, x, y, crop_size, m_im_size)
        trimap = safe_crop(trimap, x, y, crop_size, m_im_size)

        # 对每个训练对(image, trimap) 随机镜像处理.
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            fg = np.fliplr(fg)
            bg = np.fliplr(bg)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # RGB
        transform_img = transforms.ToPILImage()(img)
        transform_img = self.transform(transform_img)
        
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)
        alpha = torch.from_numpy(alpha.astype(np.float32) / 255.)
        alpha.unsqueeze_(dim=0)
        trimap = torch.from_numpy(trimap.astype(np.float32))
        trimap.unsqueeze_(dim=0)

        return transform_img, img, fg, bg, alpha, trimap

    def __len__(self):
        return len(self.names)
