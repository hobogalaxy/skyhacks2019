import csv
import logging
import os
from typing import Tuple

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_task_2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_task_1
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_task_3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from task_1 import labels_house, labels_dining_room, labels_kitchen, labels_living_room, labels_bathroom, labels_bedroom

__author__ = 'ING_DS_TECH'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = "datasets/test_dataset"
answers_file = "file.csv"

labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

labels_task2 = ['bathroom', 'bedroom', 'dinning_room', 'house','kitchen', 'living_room']

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]

output = []

model_task_1_bathroom = load_model('models/model_task_1_bathroom.h5')
model_task_1_kitchen = load_model('models/model_task_1_kitchen.h5')
model_task_1_bedroom = load_model('models/model_task_1_bedroom.h5')
model_task_1_living_room = load_model('models/model_task_1_living_room.h5')
model_task_1_dining_room = load_model('models/model_task_1_dining_room.h5')
model_task_1_house = load_model('models/model_task_1_house.h5')

model_task_2 = load_model('models/model_task_2.h5')
model_task_3_1 = load_model('models/model_task_3_1.h5')
model_task_3_2 = load_model('models/model_task_3_2.h5')


def load_image(file_path):
    img = load_img(file_path).resize((200, 200))
    return np.array(img).reshape((-1, 200, 200, 3))


preds_dict = {label: {'sum': 0, 'count': 0} for label in labels_task_1}


def generate_preds(model, labels, img):
    predicted = model.predict(img, verbose=1)
    for label, pred in zip(labels, predicted[0]):
        preds_dict[label]['sum'] += float(pred)
        preds_dict[label]['count'] += 1


def task_1(partial_output: dict, file_path: str) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))

    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    img = load_image(file_path)
    img = preprocess_input_task_1(img)

    generate_preds(model_task_1_bathroom, labels_bathroom, img)
    generate_preds(model_task_1_bedroom, labels_bedroom, img)
    generate_preds(model_task_1_dining_room, labels_dining_room, img)
    generate_preds(model_task_1_house, labels_house, img)
    generate_preds(model_task_1_kitchen, labels_kitchen, img)
    generate_preds(model_task_1_living_room, labels_living_room, img)

    for i, label in enumerate(labels_task_1):
        predicted = preds_dict[label]['sum'] / float(preds_dict[label]['count'])
        partial_output[label] = np.rint(predicted).astype(int)
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(file_path: str) -> str:
    logger.debug("Performing task 2 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    img = load_image(file_path)
    img = preprocess_input_task_2(img)
    predicted_float = model_task_2.predict(img)
    predicted = np.rint(predicted_float).astype(int)
    predicted_id = np.argmax(predicted[0])
    #
    #
    logger.debug("Done with Task 2 for file {0}".format(file_path))
    return labels_task2[predicted_id]


def task_3(file_path: str) -> Tuple[str, str]:
    logger.debug("Performing task 3 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    img = load_image(file_path)
    img = preprocess_input_task_3(img)
    predicted_1 = model_task_3_1.predict(img, verbose=1)
    predicted_1 = labels_task3_1[np.argmax(predicted_1[0])]

    predicted_2 = model_task_3_2.predict(img, verbose=1)
    predicted_2 = labels_task3_2[np.argmax(predicted_2[0])]
    #
    #
    logger.debug("Done with Task 3 for file {0}".format(file_path))
    return predicted_1, predicted_2


def main():
    logger.debug("Sample answers file generator")
    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                output_per_file = {'filename': f,
                                   'task2_class': task_2(file_path),
                                   'tech_cond': task_3(file_path)[1],
                                   'standard': task_3(file_path)[0]
                                   }
                output_per_file = task_1(output_per_file, file_path)

                output.append(output_per_file)

    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
