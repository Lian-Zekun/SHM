import logging
import os
import random
from math import isnan

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import cv2 as cv

from config import device, checkpoint_path, optimizer_, lr, mom, weight_decay, batch_size, \
                num_workers, end_epoch, decay_step, print_freq, decay_rate
from datasets import TnetDataset, SHMDataset
from models import get_t_net_model, get_m_net_model, get_shm_model


net_type = 2

def get_model():
    global net_type
    if net_type == 0:
        model = get_t_net_model()
    elif net_type == 1:
        model = get_m_net_model(4)
    elif net_type == 2:
        model = get_shm_model()
    return model
    

def get_dataset():
    global net_type
    if net_type == 0:
        dataset = TnetDataset()
    elif net_type == 1:
        dataset = SHMDataset()
    elif net_type == 2:
        dataset = SHMDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
    
    
def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
    
       
def m_net_loss(img, alpha_gt, trimap_gt, alpha_pre, epsilon_sqr=1e-12):
    mask = torch.zeros(trimap_gt.shape)
    mask[trimap_gt == 128] = 1.
    mask = mask.to(device)
    unknown_region_size = mask.sum() + epsilon_sqr
    unknown_region_size.to(device)

    # alpha diff
    alpha_loss = torch.sqrt((alpha_pre - alpha_gt) ** 2 + epsilon_sqr)
    alpha_loss = (alpha_loss * mask).sum() / unknown_region_size

    # composite rgb loss
    fg_gt = alpha_gt * img
    fg_gt.to(device)
    fg_pre = alpha_pre * img
    fg_pre.to(device)
    comp_loss = torch.sqrt((fg_pre - fg_gt) ** 2 + epsilon_sqr) / 255.
    comp_loss = (comp_loss * mask).sum() / unknown_region_size / 3.

    return 0.5 * alpha_loss + 0.5 * comp_loss
    
    
def t_net_loss(trimap_pre, trimap_gt):
    criterion = nn.CrossEntropyLoss()
    loss_t = criterion(trimap_pre, trimap_gt.long())
    return loss_t


def save_checkpoint(epoch, model, optimizer, loss, is_best, lr_decays_change, net_type):
    names = ['t', 'm', 'shm']
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lr_decays_change': lr_decays_change}
    if is_best:
        filename = '/root/checkpoint/{}_net_best_checkpoint.tar'.format(names[net_type])
        torch.save(state, filename)
    filename = '/root/checkpoint/{}_net_checkpoint.tar'.format(names[net_type])
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

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * shrink_factor


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def train_net():
    global net_type
    checkpoint = checkpoint_path[net_type]
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter(log_dir='/output')
    lr_decays_change = 0 # learning rate 下降参数

    # 加载 model
    model = get_model()
    model = nn.DataParallel(model) # 数据并行处理

    # 目前Adam是快速收敛且常被使用的优化器。随机梯度下降(SGD)虽然收敛偏慢，但是加入动量Momentum可加快收敛，
    # 同时带动量的随机梯度下降算法有更好的最优解，即模型收敛后会有更高的准确性。通常若追求速度则用Adam更多。
    if optimizer_ == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    if checkpoint:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['loss']
        lr_decays_change = ckpt['lr_decays_change']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    train_loader = get_dataset()
    print('数据加载完成!')

    # Epochs
    for epoch in range(start_epoch, end_epoch):
        # One epoch's training
        if net_type == 0:
            train_loss = t_net_train(train_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            logger=logger)
        elif net_type == 1:
            train_loss = m_net_train(train_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            logger=logger)
        elif net_type == 2:
            train_loss = shm_train(train_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            logger=logger)
                            
        if isnan(train_loss):
            print('loss nan!')
            break
        
        effective_lr = get_learning_rate(optimizer)

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # 改变 learning rate
        if epoch % 3 == 0:
            lr_decays_change += 1
            adjust_learning_rate(optimizer, decay_rate ** lr_decays_change)
            print("\nLearning_Rate change: {} * {} = {}\n".format(1e-4, lr_decays_change, 1e-4 * decay_rate ** lr_decays_change))

        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)
        save_checkpoint(epoch, model, optimizer, train_loss, is_best, lr_decays_change, net_type)
        
        
        
def t_net_train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, trimap_gt) in enumerate(train_loader):
        img = Variable(img).to(device)  # [N, 400, 400]
        trimap_gt = Variable(trimap_gt).to(device)  # [N, 400, 400]

        optimizer.zero_grad()
        trimap_pre = model(img)  # [N, 3, 320, 320]

        loss = t_net_loss(trimap_pre, trimap_gt)

        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        
        if losses.avg == 'nan':
            break

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
            logger.info(status)

    return losses.avg


def m_net_train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (transform_img, img, fg, bg, alpha, trimap) in enumerate(train_loader):
        # Move to GPU, if available
        transform_img = Variable(transform_img).to(device)  # [N, 3, 320, 320]
        img = Variable(img).to(device)  # [N, 3, 320, 320]
        fg = Variable(fg).to(device)  # [N, 3, 320, 320]
        bg = Variable(bg).to(device)  # [N, 3, 320, 320]
        alpha = Variable(alpha).to(device)  # [N, 1, 320, 320]
        trimap = Variable(trimap).to(device)  # [N, 1, 320, 320]

        # 输入 [N, 4, 320, 320],输出 [N, 1, 320, 320]
        alpha_out = model(torch.cat((transform_img, trimap / 255.), 1))
        # alpha_out = Variable(alpha_out).to(device)
        
        loss = m_net_loss(img, alpha, trimap, alpha_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        
        if isnan(losses.val):
            break

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
            logger.info(status)

    return losses.avg
    
    
def shm_train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (transform_img, img, alpha_gt, trimap_gt, mask) in enumerate(train_loader):
        # Move to GPU, if available
        transform_img = Variable(transform_img).to(device)
        img = Variable(img).to(device)  # [N, 3, 320, 320]
        alpha_gt = Variable(alpha_gt).to(device)  # [N, 1, 320, 320]
        trimap_gt = Variable(trimap_gt).to(device)  # [N, 1, 320, 320]
        mask = Variable(mask).to(device)  # [N, 320, 320]

        # 输入 [N, 4, 320, 320],输出 [N, 1, 320, 320]
        trimap_pre, alpha_pre = model(transform_img)
        # alpha_out = Variable(alpha_out).to(device)
        
        t_loss = t_net_loss(trimap_pre, mask)
        m_loss = m_net_loss(img, alpha_gt, trimap_gt, alpha_pre)
        loss = m_loss + 0.01 * t_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        
        if isnan(losses.val):
            break

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     't_loss {3} m_loss {4}\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), 
                                                                     t_loss.item(), m_loss.item(), loss=losses)
            logger.info(status)

    return losses.avg


if __name__ == '__main__':
    train_net()
