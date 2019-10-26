from keras_preprocessing.image import load_img
import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np
import csv
import os


bathroom_path = "data/main_task_data/bathroom"
bedroom_path = "data/main_task_data/bedroom"
dining_room_path = "data/main_task_data/dinning_room"
house_path = "data/main_task_data/house"
kitchen_path = "data/main_task_data/kitchen"
living_room_path = "data/main_task_data/living_room"

DIMENSIONS = (200, 200)


def load_images(path):
    image_list = []
    image_names = []
    for filename in glob.glob(path + "/*.jpg"):
        img = load_img(filename)

        img = img.resize(DIMENSIONS)  # moze tu lepszy bedzie PCA
        img = np.asarray(img)

        image_list.append(img)
        image_names.append(os.path.basename(filename))

    return image_list, image_names


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

    images = np.array(images_list)
    labels = np.array(label_list)

    return images, labels


def save(images, labels):
    with open('datasets/images' + str(DIMENSIONS[0]) + 'x' + str(DIMENSIONS[1]), 'wb') as f:
        pickle.dump(images, f)
    with open('datasets/labels', 'wb') as f:
        pickle.dump(labels, f)


def load_data():
    with open('datasets/images', 'rb') as f:
        images = pickle.load(f)
    with open('datasets/labels', 'rb') as f:
        labels = pickle.load(f)
    return images, labels


def main():

    bathrooms, bathroom_names = load_images(bathroom_path)
    bedrooms, bedroom_names = load_images(bedroom_path)
    dining_rooms, dining_room_names = load_images(dining_room_path)
    houses, house_names = load_images(house_path)
    kitchens, kitchen_names = load_images(kitchen_path)
    living_rooms, living_room_names = load_images(living_room_path)

    all_images = bathrooms + bedrooms + dining_rooms + houses + kitchens + living_rooms
    all_names = bathroom_names + bedroom_names + dining_room_names + house_names + kitchen_names + living_room_names

    label_dict = load_label_dict('labels.csv')

    images, labels = adjust_order(all_images, all_names, label_dict)

    save(images, labels)

    print(len(images), len(labels))

    # plt.imshow(images[1400, :])
    # plt.show()


main()
