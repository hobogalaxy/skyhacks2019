from keras_preprocessing.image import load_img
import glob
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import time


def load_images(path):
    image_list = []
    for filename in glob.glob(path + "/*.jpg"):
        img = load_img(filename)

        img = img.resize((200, 200))

        img = np.asarray(img)

        image_list.append(img)

    images = np.array(image_list)
    print(images.shape)
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


# now = time.time()
# augmentate("data/main_task_data/bathroom/",  "data/data_augmented/bathroom", 5)
# augmentate("data/main_task_data/bedroom/",  "data/data_augmented/bedroom", 5)
augmentate("data/main_task_data/dinning_room/",  "data/data_augmented/dinning_room", 5)
augmentate("data/main_task_data/house/",  "data/data_augmented/house", 5)
augmentate("data/main_task_data/kitchen",  "data/data_augmented/kitchen", 5)
augmentate("data/main_task_data/living_room/",  "data/data_augmented/living_room", 5)
# end = time.time()

# print(str(end - now) + "s")
