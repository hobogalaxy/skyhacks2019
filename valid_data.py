import glob
import pickle
import numpy as np
import csv
import os
from keras_preprocessing.image import load_img


def save(images, labels):
    with open('data/generated_data/validation200x200', 'wb') as f:
        pickle.dump(images, f)
    with open('data/generated_data/validation_labels', 'wb') as f:
        pickle.dump(labels, f)


def load_label_dict(path):

    label_dict = {}
    with open(path, mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            label_dict[row[0]] = row

    return label_dict


def adjust_order(images_list, image_names, labels):
    label_list = []
    images_to_remove = []
    i = 0
    for img in image_names:
        try:
            label = labels[img]
            label_list.append(label)
        except KeyError:
            images_to_remove.append(i)

        i += 1

    for index in images_to_remove:
        del images_list[index]
        for j in range(len(images_to_remove)):
            images_to_remove[j] -= 1

    images = np.array(images_list)
    labels = np.array(label_list)

    return images, labels

def load_images(path):
    image_list = []
    image_names = []
    for filename in glob.glob(path + "/*.jpg"):
        img = load_img(filename)

        img = img.resize((200, 200))  # moze tu lepszy bedzie PCA
        img = np.asarray(img)

        image_list.append(img)
        image_names.append(os.path.basename(filename))

    return image_list, image_names


def main():
    print("OK")
    images, names = load_images("data/main_task_data/validation")

    print("OK")
    label_dict = load_label_dict('labels.csv')
    print("OK")

    images, labels = adjust_order(images, names, label_dict)
    print("OK")
    save(images, labels)
    print(len(images), len(labels))
    print("OK")



main()



