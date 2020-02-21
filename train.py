from math import isnan
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from config import device, net_type, checkpoint_path, optimizer_, lr, mom, weight_decay, batch_size, \
                num_workers, end_epoch, decay_step, print_freq, decay_rate
from utils import *
from datasets import TnetDataset, MnetDataset
from models import get_t_net_model, get_m_net_model, get_shm_model


def get_model():
    if net_type == 0:
        model = get_t_net_model()
    elif net_type == 1:
        model = get_m_net_model(4)
    elif net_type == 2:
        model = get_shm_model()
    return model
    

def get_dataset():
    if net_type == 0:
        dataset = TnetDataset()
    elif net_type == 1:
        dataset = MnetDataset()
    elif net_type == 2:
        dataset = TnetDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


def train_net():
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    checkpoint = checkpoint_path[net_type]
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    lr_decays_change = 0 # learning rate 下降参数

    # 加载 model
    if checkpoint is None:
        model = get_model()
        model = nn.DataParallel(model) # 数据并行处理

        # 目前Adam是快速收敛且常被使用的优化器。随机梯度下降(SGD)虽然收敛偏慢，但是加入动量Momentum可加快收敛，
        # 同时带动量的随机梯度下降算法有更好的最优解，即模型收敛后会有更高的准确性。通常若追求速度则用Adam更多。
        if optimizer_ == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

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
                            
        if isnan(train_loss):
            print('loss nan!')
            break
        
        effective_lr = get_learning_rate(optimizer)

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # 改变 learning rate
        is_best = False
        if epoch > 0 and epoch % decay_step == 0:
            is_best = valid_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            lr_decays_change += 1
            adjust_learning_rate(optimizer, decay_rate ** lr_decays_change)
            print("\nDecays since last improvement: %d\n" % (lr_decays_change,))
        
        save_checkpoint(epoch, model, optimizer, best_loss, is_best)
        
        
def t_net_train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, trimap_gt) in enumerate(train_loader):
        img = Variable(img).to(device)  # [N, 400, 400]
        trimap_gt = Variable(trimap_gt).to(device)  # [N, 400, 400]

        trimap_pre = model(img)  # [N, 3, 320, 320]

        loss = t_net_loss(trimap_pre, trimap_gt)

        optimizer.zero_grad()
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
        
        optimizer.zero_grad()

        # 输入 [N, 4, 320, 320],输出 [N, 1, 320, 320]
        alpha_out = model(torch.cat((transform_img, trimap / 255.), 1))

        loss = m_net_loss(img, alpha, fg, bg, trimap, alpha_out)

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


if __name__ == '__main__':
    train_net()
