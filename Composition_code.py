import math
import time
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import tqdm
from tqdm import tqdm

from config import fg_path, a_path, bg_path, out_path, fg_path_test, a_path_test, bg_path_test, test_path, \
                    training_fg_names_path, training_bg_names_path, test_fg_names_path, test_bg_names_path, data_names_path
                    

comp_type = 'train'
num_bgs = {'train': 100, 'test': 20}


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
    
    with open(data_names_path, 'w') as file:
        file.write('\n'.join(names))


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.  # 对应mask图片生成 0 1 np
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(im_name, bg_name, fcount, bcount):
    global comp_type
    if comp_type == 'test':
        fg_path = fg_path_test
        a_path = a_path_test
        bg_path = bg_path_test
        out_path = test_path
        
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    bg = cv.imread(bg_path + bg_name)
    h, w = im.shape[:2]
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1: # 如果背景小于前景，放大背景，双线性插值
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out = composite4(im, bg, a, w, h)
    filename = out_path + str(fcount) + '_' + str(bcount) + '.png'
    cv.imwrite(filename, out)


def process_one_fg(fcount):
    global num_bgs, comp_type, bg_files
    im_name = fg_files[fcount]
    bcount = fcount * num_bgs[comp_type]

    for i in range(num_bgs):
        bg_name = bg_files[bcount]
        process(im_name, bg_name, fcount, bcount)
        bcount += 1


def do_composite():
    print('Doing composite training data...')
    global num_bgs, comp_type, fg_files
    num_samples = len(fg_files) * num_bgs[comp_type]
    print('num_samples: ' + str(num_samples))

    start = time.time()

    with Pool(processes=16) as p:
        max_ = len(fg_files)
        print('num_fg_files: ' + str(max_))
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(process_one_fg, range(0, max_)))):
                pbar.update()

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))
    

if __name__ == '__main__':
    global bg_files, fg_files
    with open(training_bg_names_path) as f:
        bg_files = f.read().splitlines()
    with open(training_fg_names_path) as f:
        fg_files = f.read().splitlines()

    do_composite()
    
    with open(test_bg_names_path) as f:
        bg_files = f.read().splitlines()
    with open(test_fg_names_path) as f:
        fg_files = f.read().splitlines()
        
    do_composite()
    
    gen_names()
    

