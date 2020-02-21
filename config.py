import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# 训练参数
net_type = 1  # 网络类型 0: t_net, 1: m_net, 2: shm
end_epoch = 1000  # 迭代次数
lr = 1e-4  # learning rate
decay_rate = 0.9  # lr 衰减指数
decay_step = 20  # learning rate 下降周期
optimizer_ = 'Adam'  # 优化器
weight_decay = 0.0005  # L2 权重衰减
mom = 0.9  # SGD momentum
batch_size = 2  # 批次大小
num_workers = 4  # DataLoader 进程数
print_freq = 100  # 打印周期

# MNet 图片大小
m_im_size = 320
# TNet 图片大小
t_im_size = 400
# shm 图片大小
s_im_size = 800

num_samples = 43100  # 数据集总数
num_train = 34480  # 训练集数量
num_valid = 8620  # 测试集数量

##############################################################
# paths

checkpoint_path = [None, None, None]

fg_path = '../data/fg/'
a_path = '../data/alpha/'
bg_path = '../data/bg/'
out_path = '../data/merged/'
trimap_path = '../data/trimap/'

training_fg_names_path = '../data/Combined_Dataset/Training_set/training_fg_names.txt'
training_bg_names_path = '../data/Combined_Dataset/Training_set/training_bg_names.txt'
data_names_path = 'names.txt'

max_size = 1600
fg_path_test = '../data/fg_test/'
a_path_test = '../data/alpha_test/'
bg_path_test = '../data/bg_test/'
out_path_test = '../data/merged_test/'
trimap_path_test = '../data/trimap_test/'
##############################################################
