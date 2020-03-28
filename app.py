import io
import os

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from flask import Flask, request, render_template
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
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True

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
    

def generate_alpha(img, model):
    img = Image.open(io.BytesIO(img))
    img.save('uploads/1.png')
    if img.mode != 'RGB':
        img = img.convert("RGB")    
    img = transformer(img)
    img.unsqueeze_(dim=0)
    Variable(img).to(device)
    
    with torch.no_grad():
        _, alpha = model(img)
        
    alpha = alpha.cpu().data.numpy()
    alpha = alpha[0][0] * 255.0
    alpha = alpha.astype(np.uint8)

    return alpha


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':            
        if request.files.get('image'):
            image = request.files['image']
            image_bytes = image.read()
            alpha = generate_alpha(image_bytes, model)
            
            if image and allowed_file(image.filename):
                name = secure_filename(image.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], name)
                cv.imwrite(path, alpha)
                print('图片生成')
                return render_template('index.html', path=path)
        else:
            flash('No file part')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    load_model()
    app.run()