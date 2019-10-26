import datetime
import pickle

import tensorflow as tf
from keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

import resnet

from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

NUM_GPUS = 1
BS_PER_GPU = 128
NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def load_data():
    with open('data/generated_data/images200x200', 'rb') as f:
        images = pickle.load(f)
    with open('data/generated_data/labels', 'rb') as f:
        labels = pickle.load(f)
    return images, labels


images, labels = load_data()

print(len(images), len(labels))

labels = labels[:, 2]


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)


# from sklearn.utils import shuffle
# x, y = shuffle(x, y, random_state=0)

# x, y = train_test_split(images, test_size=0.8)
# test_x, test_y = train_test_split(housing_cat_1hot, test_size=0.8)


input_shape = (200, 200, 3)
# img_input = tf.keras.layers.Input(shape=input_shape)
# opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')

x = base_model.output
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(6, activation='relu')(x)
x = tf.keras.layers.Activation("softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)


for layer in base_model.layers:
    layer.trainable = False

model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

model.fit(x=images, y=labels, epochs=60, batch_size=128, validation_split=0.1, shuffle=True)


# model = resnet.resnet56(img_input=img_input, classes=6)

# model.evaluate(test_dataset)

model.save('model.h5')

# new_model = keras.models.load_model('model.h5')

