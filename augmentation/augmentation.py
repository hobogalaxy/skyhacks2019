from keras_preprocessing.image import load_img
import glob
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import time


def load_images(path):
    image_list = []
    for filename in glob.glob(path + "/*.jpg"):
        img = load_img(filename)

        img = img.resize((300, 300))

        img = np.asarray(img)

        image_list.append(img)

    images = np.array(image_list)
    print(images.shape)
    # images = np.expand_dims(images, axis=0)
    return images


def augmentate(path, save_dir, times):
    aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    images = load_images(path)

    generator = aug.flow(images, batch_size=len(images), save_to_dir=save_dir,
                         save_prefix="image", save_format="jpg")

    total = 0
    for i in generator:
        total += 1
        if total == times:
            break


# now = time.time()
# augmentate("/home/adrian/skyhacks/data/main_task_data/bathroom/",  "/home/adrian/skyhacks/test/", 1)
# end = time.time()

# print(str(end - now) + "s")
