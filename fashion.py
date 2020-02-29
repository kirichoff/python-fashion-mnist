import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

class_names = [i.strip() for i in open("./clases.dat").readlines()]

img_raw = tf.io.read_file('./t-shirt.jpg')
img_tensor = tf.image.decode_jpeg(img_raw, channels=1)
img = tf.image.resize(img_tensor, [28, 28])
# plt.imshow(img)
# plt.show()
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
img = (np.expand_dims(img, 0))

text_image = (np.expand_dims(test_images[1156], 0))
text_image = test_images[1156]

plt.imshow(text_image, cmap=plt.cm.binary)
# plt.show()


# image = Image.open('./images.jfif')

# inverted_image = ImageOps.invert(image)

# inverted_image.save('./dr1.png')

test_image = image.load_img('./t-shirt.jpg', target_size=(28, 28), color_mode='grayscale')
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(28, 28)


def loadimage(path):
    im = image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    im = image.img_to_array(im)
    im /= 255.0
    im = im.reshape(28, 28)
    im = np.expand_dims(im, axis=0)
    return im


plt.imshow(test_image, cmap=plt.cm.binary)
# plt.show()
test_image = np.expand_dims(test_image, axis=0)

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1000)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

    # Plot the first X test images, their predicted labels, and the true labels.


# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
print("------------------------------------------------")

pathesimg_pathes = ['./t-shirt.jpg', './t-shirt2.png', './dr1.png', './dr.png', './images.jfif', './dress.jpg',
                    './shoes.jpg']

for key in pathesimg_pathes:
    print(key)
    result = loadimage(key)
    predictions_single = model.predict(result)
    predicted_label = np.argmax(predictions_single)
    print(class_names[predicted_label], predictions_single, np.argmax(predictions_single))
