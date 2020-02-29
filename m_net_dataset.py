import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import m_im_size, fg_path, bg_path, a_path, trimap_path, out_path, training_fg_names_path, training_bg_names_path

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
def safe_crop(mat, x, y, crop_size, im_size):
    if len(mat.shape) == 2:
        ret = np.zeros((crop_size, crop_size), np.uint8)
    else:
        ret = np.zeros((crop_size, crop_size, 3), np.uint8)
    crop = mat[y:y + crop_size, x:x + crop_size]
    h, w = crop.shape[:2] # 若原图片小于目标大小,则crop shape不会保证预期的crop_size,需要重新得到shape
    ret[0:h, 0:w] = crop
    if crop_size != im_size:
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret


class DIMDataset(Dataset):
    def __init__(self):
        with open('names.txt', 'r') as file:
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
        
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)  # RGB
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)  # RGB
        alpha = torch.from_numpy(alpha.astype(np.float32) / 255.)
        alpha.unsqueeze_(dim=0)
        # mask = np.equal(trimap, 128).astype(np.float32)
        # mask = torch.from_numpy(mask)
        trimap = torch.from_numpy(trimap.astype(np.float32))
        trimap.unsqueeze_(dim=0)
        # transform_img = torch.cat((transform_img, trimap / 255.), 1)

        return transform_img, img, fg, bg, alpha, trimap

    def __len__(self):
        return len(self.names)
