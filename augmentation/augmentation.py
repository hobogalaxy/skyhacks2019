from keras_preprocessing.image import load_img
import glob
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from matplotlib import pyplot as plt


def load_images(path):
    image_list = []
    for filename in sorted(glob.glob(path + "/*.jpg")):
        img = load_img(filename)

        img = img.resize((300, 300))

        img = np.asarray(img)

        image_list.append(img)
    images = np.array(image_list)
    # images = np.expand_dims(images, axis=0)
    return images


def augmentate(path, save_dir, new_img_num):
    aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=(0.5, 1.5))

    images = load_images(path)

    generator = aug.flow(images, batch_size=len(images), save_to_dir=save_dir,
                         save_prefix="image", save_format="jpg")

    total = 0
    for i in generator:
        total += 1
        if total == new_img_num:
            break

#


def numer(nazwa):
    return nazwa.split("_")[1]

def sortowanie(lista_nazw):
    slownik = {}
    for i in range(len(lista_nazw)):
        slownik[lista_nazw[i]] = int(numer(lista_nazw[i]))
    sorted_slownik = sorted(slownik.items(), key=lambda kv: kv[1])
    return list(i[0] for i in sorted_slownik)



def label_gen(directory_of_augmented_pics, path_to_dataset_pics):
    dataset_dataframe = pd.read_csv('./labels.csv')
    licznik = 0
    new_labels = pd.DataFrame(columns=dataset_dataframe.columns)
    for filename in sorted(glob.glob(path_to_dataset_pics + "/*.jpg")):
        for aug_filename in sortowanie(glob.glob(directory_of_augmented_pics + "/*.jpg"))[5 * licznik: 5 * licznik + 5]:
            wiersz = dataset_dataframe.loc[dataset_dataframe['filename'] == filename[-44:]]
            wiersz = wiersz.set_value(wiersz.index, "filename", aug_filename[8:])
            new_labels = new_labels.append(wiersz)
        licznik += 1
    new_labels.reset_index()
    new_labels.to_csv("new_labels.csv")


#augmentate("../dataset/house/",  "../test/", 5)
label_gen("../test/", "../dataset/house/")