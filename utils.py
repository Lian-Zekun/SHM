import argparse
import logging
import os
import random

import cv2 as cv
import numpy as np
import torch

from config import num_valid


def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]
    
    with open('names.txt', 'w') as file:
        file.write('\n'.join(names))

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint/t_net_checkpoint.pth'
    torch.save(state, filename)
    if is_best:
        filename = 'checkpoint/t_net_best_checkpoint.pth'
        torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end_epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='start learning rate')
    parser.add_argument('--decay_step', type=int, default=20, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--pretrained', type=bool, default=True, help='pretrained model')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


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


# alpha-prediction loss 是对每个像素的 
# groundtruth alpha 值与 predicted alpha 值间的绝对差值(absolute difference).
# 但由于绝对值的不可微，故采用其逼近形式
def alpha_prediction_loss(alpha_pred, alpha, trimap, epsilon_sqr=1e-12):
    diff = alpha_pred - alpha
    diff = diff * trimap
    num_pixels = torch.sum(trimap) + epsilon_sqr
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / num_pixels


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse(pred, alpha, trimap):
    num_pixels = float((trimap == 128).sum())
    return ((pred - alpha) ** 2).sum() / num_pixels


# compute the SAD error given a prediction and a ground trputh.
#
def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
