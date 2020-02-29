import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, decay_rate
from t_net_dataset import TNetDataset
from utils import parse_args, save_checkpoint, AverageMeter, get_logger, get_learning_rate, \
    alpha_prediction_loss, adjust_learning_rate
    
    
def get_t_net_model():
    model = smp.PSPNet('resnet50', classes=3)
    # for p in model.encoder.parameters():
        # p.requires_grad = False
    return model
    

def loss_function(trimap_pre, trimap_gt):
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt.long())
    return L_t


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
        model = get_t_net_model()
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

    model = model.to(device)

    train_dataset = TNetDataset()
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
        if train_loss == 'nan':
            print('loss nan!')
            break
        
        effective_lr = get_learning_rate(optimizer)

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        if epoch > 0 and epoch % args.decay_step == 0:
            is_best = train_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            lr_decays_change += 1
            adjust_learning_rate(optimizer, decay_rate ** lr_decays_change)
            print("\nDecays since last improvement: %d\n" % (lr_decays_change,))
        
        save_checkpoint(epoch, model, optimizer, best_loss, is_best, 0)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, trimap_gt) in enumerate(train_loader):
        img = Variable(img).to(device)  # [N, 400, 400]
        trimap_gt = Variable(trimap_gt).to(device)  # [N, 400, 400]

        trimap_pre = model(img)  # [N, 3, 320, 320]

        loss = loss_function(trimap_pre, trimap_gt)

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


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
