from keras_preprocessing.image import load_img
import glob
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from matplotlib import pyplot as plt


def load_images(path):
    image_list = []
    for filename in sorted(glob.glob(path + "/*.jpg")):
        img = load_img(filename)

        img = img.resize((300, 300))

        img = np.asarray(img)

        image_list.append(img)
    images = np.array(image_list)
    print(images.shape)
    # images = np.expand_dims(images, axis=0)
    plt.imshow(images[0])
    plt.show()
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
# augmentate("../dataset/bedroom/",  "../test/", 5)
#
# # print(str(end - now) + "s")
my_directory = "../test/"
data = pd.read_csv('./labels.csv')
path = "../dataset/bedroom/"
licznik = 0
dictionary = {}
for filename in sorted(glob.glob(path + "/*.jpg")):
    for aug_filename in sorted(glob.glob(my_directory + "/*.jpg"))[5 * licznik: 5 * licznik + 5]:
            dictionary[aug_filename[8:]] = data.loc[data['filename'] == filename[-44:]]
            # print(pd.DataFrame(data.loc[data['filename'] == filename[-44:]]))
    licznik += 1


print(dictionary)