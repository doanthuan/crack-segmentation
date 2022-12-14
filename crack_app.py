import os
import random
import string
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import predict_image

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#model = load_unet_vgg16(os.path.join(BASE_DIR , 'models/model_unet_vgg_16_best.pt'))
model = tf.keras.models.load_model('models/saved_model/unet')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

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
            
        image = predict_image(model, filepath)
        im = Image.fromarray(image)

        file_result = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) + ".jpg"

        im.save(os.path.join(BASE_DIR, 'results', file_result))
        
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