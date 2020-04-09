from keras.preprocessing import image
import cv2
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def predict(model, picture):
    img=cv2.imread(picture)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))

    plt.imshow(img)
    img = np.reshape(img,[1,28,28,1])

    pred = model.predict(img)
    return np.argmax(pred,axis=1)

picture = '3.png'
model = tf.keras.models.load_model('model_mnist_cnn.h5')
predict(model, picture)