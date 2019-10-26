import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model

from sklearn.metrics import classification_report

from os import path
import pickle
import numpy as np

# DATA_PATH = 'datasets/100x100/'
DATA_PATH = 'datasets/200x200/'

with open(path.join(DATA_PATH, 'images'), 'rb') as f:
    all_images = pickle.load(f)

with open(path.join(DATA_PATH, 'labels'), 'rb') as f:
    all_targets = pickle.load(f)

with open(path.join(DATA_PATH, 'validation'), 'rb') as f:
    validation_images = pickle.load(f)

with open(path.join(DATA_PATH, 'validation_labels'), 'rb') as f:
    validation_targets = pickle.load(f)

all_images = np.concatenate([all_images, validation_images], axis=0)
all_targets = np.concatenate([all_targets, validation_targets], axis=0)

all_images = np.array(all_images)
all_images = preprocess_input(all_images)

all_targets = all_targets[:, 4:]
all_targets = all_targets.astype(int)

labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

labels_bathroom = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Plumbing fixture', 'Shower',
                   'Sink', 'Tap', 'Toilet', 'Tile']

labels_kitchen = ['Cabinetry', 'Countertop', 'Drawer', 'Sink', 'Kitchen & dining room table',
                  'Refrigerator', 'Kitchen', 'Kitchen stove', 'Tile']

labels_bedroom = ['Bed', 'Bed frame', 'Bed sheet', 'Bedroom', 'Chest of drawers', 'Drawer', 'Mattress',
                  'Nightstand', 'Curtain']

labels_living_room = ['Cupboard', 'Living room', 'Chandelier', 'Chair', 'Coffee table', 'Fireplace',
                      'Nightstand', 'Couch', 'Curtain']

labels_dining_room = ['Dining room', 'Chair', 'Tablecloth', 'Table']

labels_house = ['Ceiling', 'Door', 'Facade',  'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Property',
                'Real estate',  'Roof', 'Room', 'Rural area', 'Sky', 'Tree', 'Urban area', 'Wall', 'Window']


def prepare_model(labels):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling='avg')

    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(labels), activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train(images, targets, labels, model_name):
    # Configure training
    model = prepare_model(labels)

    checkpoint = ModelCheckpoint('models/model_task_1_' + model_name + '.h5', monitor='val_loss', verbose=1,
                                 mode='min', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
    callbacks_list = [checkpoint, reduce_lr]

    # Load subset of targets
    labels_indices = [labels_task_1.index(label) for label in labels]
    targets = np.array([[targets[row, index] for index in labels_indices] for row in range(targets.shape[0])])

    # Shuffle
    randomize = np.arange(len(images))
    np.random.shuffle(randomize)
    images = images[randomize]
    targets = targets[randomize]

    model.fit(x=images, y=targets, epochs=5, batch_size=64, validation_split=0.2, shuffle=True, callbacks=callbacks_list)

    targets = model.predict(images)
    print(targets[0])

    scores = model.evaluate(images, targets, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def test(images, targets, labels, model_name):
    model = load_model('models/model_task_1_' + model_name + '.h5')
    predicted = model.predict(images, verbose=1)

    # Load subset of targets
    labels_indices = [labels_task_1.index(label) for label in labels]
    targets = np.array([[targets[row, index] for index in labels_indices] for row in range(targets.shape[0])])

    targets = np.rint(targets).astype(int)
    predicted = np.rint(predicted).astype(int)

    print(classification_report(targets, predicted, target_names=labels_task_1))


def train_all(images, targets):
    train(images, targets, labels_bathroom, 'bathroom')
    train(images, targets, labels_kitchen, 'kitchen')
    train(images, targets, labels_bedroom, 'bedroom')
    train(images, targets, labels_living_room, 'living_room')
    train(images, targets, labels_dining_room, 'dining_room')
    train(images, targets, labels_house, 'house')


if __name__ == "__main__":
    all_images = all_images[:5]
    all_targets = all_targets[:5]
    # train_all(all_images, all_targets)
    test(all_images, all_targets, labels_bathroom, 'bathroom')
