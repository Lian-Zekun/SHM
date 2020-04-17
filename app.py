import io
import os

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from flask import Flask, request, render_template, flash, send_from_directory, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import cv2 as cv
import numpy as np

from models import get_shm_model
from config import device, checkpoint_path


transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


modle = None
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['COMPOSITE_FOLDER'] = 'static/composite/'
app.config['DEBUG'] = False
app.config['SECRET_KEY'] = b'_5#y2L"F4Q8z\n\xec]/'

def load_model():
    global model
    model = get_shm_model()
    model = nn.DataParallel(model)
    checkpoint = checkpoint_path[2]
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print('模型加载完成')
    

def generate_alpha(path):
    global model
    img = cv.imread(path)
    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img.unsqueeze_(dim=0)
    Variable(img).to(device)
    
    with torch.no_grad():
        _, alpha = model(img)
        
    alpha = alpha.cpu().data.numpy()
    alpha = alpha[0][0] * 255.0
    alpha = alpha.astype(np.uint8)

    return alpha
    
def composite(fg, bg, h, w, a):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    cv.imwrite('static/composite/150.png', a)    
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp
    

def color_bg_composite(fg_path, rgb, a):
    fg = cv.imread(fg_path)
    h, w = fg.shape[:2]
    # 得到 rgb 各通道颜色值
    r, g, b = int(rgb[1:3], 16), int(rgb[3:5], 16), int(rgb[5:7], 16)
    bg = np.ones((h, w, 3), np.uint8)
    bg[:, :, 0] = np.ones((h, w)) * b
    bg[:, :, 1] = np.ones((h, w)) * g
    bg[:, :, 2] = np.ones((h, w)) * r
    
    #return composite(fg, bg, h, w, a)
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    cv.imwrite('static/composite/150.png', a)    
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def fg_bg_composite(fg_path, bg_path, a):
    fg = cv.imread(fg_path)
    bg = cv.imread(bg_path)
    h, w = fg.shape[:2]
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1: # 如果背景小于前景，放大背景，双线性插值
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    #return composite(fg, bg, h, w, alpha)
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    cv.imwrite('static/composite/150.png', a)    
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def composite():
    if request.method == 'POST':            
        if request.files.get('fg'):
            if not request.form.get('rgb') and not request.files.get('bg'):
                flash('请选择纯色背景或上传背景图片')
                return render_template('index.html')
            fg = request.files['fg']
            if allowed_file(fg.filename):
                fg_name = secure_filename(fg.filename)
                fg_path = os.path.join(app.config['UPLOAD_FOLDER'], fg_name)
                fg.save(fg_path)
                if request.files.get('bg'):
                    bg = request.files['bg']
                    if allowed_file(bg.filename):
                        bg_name = secure_filename(bg.filename)
                        bg_path = os.path.join(app.config['UPLOAD_FOLDER'], bg_name)
                        bg.save(bg_path)
                        alpha = generate_alpha(fg_path)
                        comp = fg_bg_composite(fg_path, bg_path, alpha)
                    else:
                        flash('文件格式必须为 png、jpg、jpeg 之一')
                        return render_template('index.html')
                elif request.form.get('rgb'):
                    rgb = request.form['rgb']
                    alpha = generate_alpha(fg_path)
                    comp = color_bg_composite(fg_path, rgb, alpha)

                path = os.path.join(app.config['COMPOSITE_FOLDER'], fg_name)
                cv.imwrite(path, comp)
                print('图片生成')
                
                return render_template('index.html', name=fg_name)
            else:
                flash('文件格式必须为 png、jpg、jpeg 之一')
                render_template('index.html')
        else:
            flash('请上传文件')
            render_template('index.html')

    return render_template('index.html')
    
    
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join(app.config['COMPOSITE_FOLDER'], filename)):
            return send_from_directory(app.config['COMPOSITE_FOLDER'], filename, as_attachment=True)
        pass

if __name__ == '__main__':
    load_model()
    app.run()