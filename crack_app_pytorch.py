from __future__ import division
import os
import random
import string
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

from crack_utils import load_unet_vgg16
import numpy as np
import torchvision.transforms as transforms
from unet.unet_transfer import input_size
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2 as cv


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_unet_vgg16(os.path.join(BASE_DIR , 'models/model_unet_vgg_16_best.pt'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

channel_means = [0.485, 0.456, 0.406]
channel_stds  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

def predict_img(model, path):

    img_0 = Image.open(str(path))
    img_0 = np.asarray(img_0)
    if len(img_0.shape) != 3:
        print(f'incorrect image shape: {path}{img_0.shape}')
        return

    img_0 = img_0[:,:,:3]

    input_width, input_height = input_size[0], input_size[1]
    img_height, img_width, img_channels = img_0.shape

    img_1 = cv.resize(img_0, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)

    file_result = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) + ".jpg"

    cv.imwrite(filename= os.path.join(BASE_DIR, 'results', file_result), img=(mask * 255).astype(np.uint8))

    return file_result

@app.route('/', methods=['GET'])
def index(name=None):
    return render_template("upload.html")

@app.route('/upload', methods=['GET','POST'])
def upload(name=None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'upload_file' not in request.files:
            return 'No file part'
        file = request.files['upload_file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # split_tup = os.path.splitext(filepath)
        # file_ext = split_tup[1]
        # if file_ext == ".mp4":
            
        file_result = predict_img(model, filepath)
        
        #display result
        return render_template("result_crack.html", filename=filename, file_result=file_result)
        
    else:
        return render_template('upload.html')


# Custom static data
@app.route('/uploads/<path:filename>')
def upload_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<path:filename>')
def result_static(filename):
    return send_from_directory('results', filename)

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=5001)