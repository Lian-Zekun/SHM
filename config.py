import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# MNet 图片大小
m_im_size = 320
# TNet 图片大小
t_im_size = 400

decay_rate = 0.9  # lr 衰减指数

num_samples = 43100  # 数据集总数
num_train = 34480  # 训练集数量
num_valid = 8620  # 测试集数量

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

channels = {
    'vgg': {
        'encoder': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                    'M', 512, 512, 512],
        'decoder': [512, 256, 128, 64, 64]
    },
    'refinement': {
        'encoder': [64, 64, 64]
    }
}

# 数据标准化
data_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

##############################################################
# Set your paths here

fg_path = '../data/fg/'
a_path = '../data/alpha/'
bg_path = '../data/bg/'
out_path = '../data/merged/'
trimap_path = '../data/trimap/'

training_fg_names_path = '../data/Combined_Dataset/Training_set/training_fg_names.txt'
training_bg_names_path = '../data/Combined_Dataset/Training_set/training_bg_names.txt'

max_size = 1600
fg_path_test = '../data/fg_test/'
a_path_test = '../data/alpha_test/'
bg_path_test = '../data/bg_test/'
out_path_test = '../data/merged_test/'
trimap_path_test = '../data/trimap_test/'
##############################################################
