import numpy as np
from keras.preprocessing import image
import tensorflow as tf

LETTER_IMAGES_FOLDER = "training_generate/divided_sample"
MODEL_FILENAME = "Fashion-model.h5"
class_names = [i.strip() for i in open("./clases.dat").readlines()]


def load_image(path):
    im = image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    im = image.img_to_array(im)
    im /= 255.0
    im = im.reshape(28, 28)
    im = np.expand_dims(im, axis=0)
    return im


new_model = tf.keras.models.load_model(MODEL_FILENAME)

test_images = ['./t-shirt.jpg', './t-shirt2.png', './dr1.png', './dr.png', './dr2.png', './images.jfif', './dress.jpg',
               './shoes.jpg']

for key in test_images:
    print(key)
    result = load_image(key)
    predictions_single = new_model.predict(result)
    predicted_label = np.argmax(predictions_single)
    print(class_names[predicted_label], predictions_single, predicted_label)
