import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template

from keras.preprocessing import image
import cv2
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'C:\images_mnist'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # # a simple page that says hello
    # @app.route('/')
    # def hello():
    #     return 'Hello, World!'

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    def predict(model, picture):
        img=cv2.imread(picture)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(28,28))

        # plt.imshow(img)
        img = np.reshape(img,[1,28,28,1])

        pred = model.predict(img)
        return np.argmax(pred,axis=1)

    @app.route("/", methods=['GET', 'POST'])
    def index():
        result = ''
        model = tf.keras.models.load_model('app/model_mnist_cnn.h5')
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                result = predict(model, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print('THIS IS RESULT ------------ ', result)
                # r = result.tolist()
                # result_str = ''.join(r)
                print(type(result))
        return render_template('index.html', result=result)

    from . import db
    db.init_app(app)

    return app