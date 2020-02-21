import argparse
import logging
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch import nn

from config import num_valid, device

############################################################################################################
# 得到所有数据的名字文件


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
        

############################################################################################################
# train 


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size in each context')
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
    
       
def m_net_loss(img, alpha, fg, bg, trimap, alpha_out, epsilon_sqr=1e-12):
    trimap_3 = torch.cat((trimap, trimap, trimap), 1).to(device)
    unknown_region_size = trimap.sum() + epsilon_sqr

    # alpha diff
    alpha_loss = torch.sqrt((alpha_out - alpha) ** 2 + epsilon_sqr)
    alpha_loss = (alpha_loss * trimap).sum() / unknown_region_size

    # composite rgb loss
    alpha_out_3 = torch.cat((alpha_out, alpha_out, alpha_out), 1)
    comp = alpha_out_3 * fg + (1. - alpha_out_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + epsilon_sqr) / 255.
    comp_loss = (comp_loss * trimap_3).sum() / unknown_region_size / 3.

    return 0.5 * alpha_loss + 0.5 * comp_loss
    
    
def t_net_loss(trimap_pre, trimap_gt):
    criterion = nn.CrossEntropyLoss()
    loss_t = criterion(trimap_pre, trimap_gt.long())
    return loss_t


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
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
    
    
############################################################################################################
# extract


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


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
