import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from config import a_path, trimap_path, trimap_path_test

num_bgs = 100


def gen_trimap(a_name, fcount, bcount):
    alpha = cv.imread(a_path + a_name, 0)
    # 对 trimaps 从其 groundtruth alpha mattes 进行随机膨胀(dilated)
    # 以使得模型对 trimap 位置更加鲁棒.
    iterations = np.random.randint(low=1, high=20)
    erode_ksize = np.random.randint(low=1, high=5)
    dilate_ksize = np.random.randint(low=1, high=5)
    erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    dilated = cv.dilate(alpha, dilate_kernel, iterations)
    eroded = cv.erode(alpha, erode_kernel, iterations)
    trimap = 128 * np.ones_like(alpha)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    
    filename = trimap_path + str(fcount) + '_' + str(bcount) + '.png'
    cv.imwrite(filename, trimap)


def process_one_fg(fcount):
    a_name = fg_files[fcount]
    bcount = fcount * num_bgs

    for i in range(num_bgs):
        gen_trimap(a_name, fcount, bcount)
        bcount += 1

    
if __name__ == '__main__':
    with open('../data/Combined_Dataset/Training_set/training_fg_names.txt') as f:
        fg_files = f.read().splitlines()
        
    num = len(fg_files)
    print('num_samples: ' + str(num * num_bgs))
    
    for fcount in tqdm(range(num)):
        process_one_fg(fcount)
        
    
    
