import os

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

from config import device, test_path, a_path_test, out_path_test, checkpoint_path, trimap_path_test
from models import get_shm_model, get_t_net_model


transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

def load_model():
    model = get_shm_model()
    # model = get_t_net_model()
    model = nn.DataParallel(model)
    checkpoint = checkpoint_path[2]
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    return model
    

def generate_alpha(img, model):   
    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img.unsqueeze_(dim=0)
    Variable(img).to(device)
    
    with torch.no_grad():
        trimap, alpha = model(img)
        
    trimap = torch.argmax(trimap[0], dim=0)
    trimap[trimap == 0] = 0
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255
    trimap_np = trimap.cpu().data.numpy()

    trimap_np = trimap_np.astype(np.uint8)
        
    alpha = alpha.cpu().data.numpy()
    alpha = alpha[0][0] * 255.0
    alpha = alpha.astype(np.uint8)

    return trimap_np, alpha


if __name__ == '__main__':
    model = load_model()
    print('模型加载完成')
    
    test_names = os.listdir(test_path)

    for name in tqdm(test_names):
        img = cv.imread(test_path + name)       
        trimap, alpha_pre = generate_alpha(img, model)
        cv.imwrite(out_path_test + name, alpha_pre)
        cv.imwrite(trimap_path_test + name, trimap)
    print('分离完毕')
