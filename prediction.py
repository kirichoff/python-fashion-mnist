import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
import tensorflow as tf

import image_fit


LETTER_IMAGES_FOLDER = "training_generate/divided_sample"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

image_file = 'test_0.bmp'
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image_fit.resize_to_fit(image, 12, 22)

new_model = tf.keras.models.load_model(MODEL_FILENAME)
predicitions = new_model.predict(image)
print(np.argmax(predicitions[0]))