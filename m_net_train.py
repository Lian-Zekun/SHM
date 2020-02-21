from math import isnan
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable

from config import device, grad_clip, print_freq, decay_rate
from m_net_dataset import DIMDataset
from m_net_model import get_m_net_model
from utils import parse_args, save_checkpoint, AverageMeter, get_logger, get_learning_rate, \
    alpha_prediction_loss, adjust_learning_rate
    
    
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


def train_net(args):
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    lr_decays_change = 0 # learning rate 下降参数

    # 加载 model
    if checkpoint is None:
        model = get_m_net_model()
        model = nn.DataParallel(model) # 数据并行处理

        # 目前Adam是快速收敛且常被使用的优化器。随机梯度下降(SGD)虽然收敛偏慢，但是加入动量Momentum可加快收敛，
        # 同时带动量的随机梯度下降算法有更好的最优解，即模型收敛后会有更高的准确性。通常若追求速度则用Adam更多。
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    print("加载数据...")
    train_dataset = DIMDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('数据加载完成!')

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
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
        if epoch > 0 and epoch % args.decay_step == 0:
            is_best = valid_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            lr_decays_change += 1
            adjust_learning_rate(optimizer, decay_rate ** lr_decays_change)
            print("\nDecays since last improvement: %d\n" % (lr_decays_change,))
        
        save_checkpoint(epoch, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    
    optimizer.zero_grad()

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


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
