import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import classification_report

from os import path
import pickle
import numpy as np

# DATA_PATH = 'datasets/100x100/'
DATA_PATH = 'datasets/200x200/'

with open(path.join(DATA_PATH, 'images'), 'rb') as f:
    images = pickle.load(f)

with open(path.join(DATA_PATH, 'labels'), 'rb') as f:
    labels = pickle.load(f)

with open(path.join(DATA_PATH, 'validation'), 'rb') as f:
    validation_images = pickle.load(f)

with open(path.join(DATA_PATH, 'validation_labels'), 'rb') as f:
    validation_labels = pickle.load(f)

print(images.shape, ' ', validation_images.shape)
images = np.concatenate([images, validation_images], axis=0)
labels = np.concatenate([labels, validation_labels], axis=0)

images = np.array(images)
images = preprocess_input(images)

labels = labels[:, 1].reshape(-1, 1)
labels = labels.astype(int)

# Ordinal Classification:
# https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c

def prepare_model():
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling='avg')

    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(3, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train(images, labels):
    model = prepare_model()

    checkpoint = ModelCheckpoint('models/model_task3_1.h5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
    callbacks_list = [checkpoint, reduce_lr]

    images = preprocess_input(images)

    def encode_ordinal(class_val):
        return [
            1 if class_val > 1 else 0,
            1 if class_val > 2 else 0,
            1 if class_val > 3 else 0
        ]
    labels = np.array(list(map(encode_ordinal, labels)))

    # Shuffle
    randomize = np.arange(len(images))
    np.random.shuffle(randomize)
    images = images[randomize]
    labels = labels[randomize]

    model.fit(x=images, y=labels, epochs=5, batch_size=32, validation_split=0.3, shuffle=True, callbacks=callbacks_list)

    # model.save('adiz_trained.h5')

    labels = model.predict(images)

    scores = model.evaluate(images, labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

def test(images, labels):
    images = images[:10]
    labels = labels[:10]
    model = load_model('models/model_task3_1.h5')
    predicted = model.predict(images, verbose=1)
    def decode_ordinal(ordinal):
        return [
            1 - ordinal[0],
            ordinal[0] - ordinal[1],
            ordinal[1] - ordinal[2],
            ordinal[2]
        ]
    predicted = list(map(decode_ordinal, predicted))
    print(labels)
    print(predicted)


train(images, labels)
# test(images, labels)
