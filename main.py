import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


VALIDATION_SPLIT = 0.4
EPOCHS = 10
BATCH_SIZE = 64
opt = keras.optimizers.Adam(learning_rate=0.001)

img_path = 'data/generated_data/images200x200'
label_path = 'data/generated_data/labels'

ORDER = [0,4,3,1,5,2,4,1,3,4,5,0,0,2,1,0,3,0,5,5,1,2,1,3,2,4,3,4,2,5]
print(len(ORDER))


def load_data():
    with open(img_path, 'rb') as f:
        images = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return images, labels



from keras_preprocessing.image import load_img
import glob
import numpy as np
def eval_test():
    new_model = keras.models.load_model('new_best_model.h5')
    test_images = []
    for filename in glob.glob("test_results_dataset/test_dataset" + "/*.jpg"):
        img = load_img(filename)
        img = img.resize((200, 200))
        img = np.asarray(img)
        # print(new_model.predict(img.reshape((1,img.shape[0],img.shape[1],3))))
        test_images.append(img.reshape((1, img.shape[0], img.shape[1], 3)))
        # plt.imshow(img)
        # plt.show()

    test_labels = []
    for img in test_images:
        test_labels.append(np.argmax(new_model.predict(img)))

    print(test_labels)
    print(ORDER)
    result = [1 if test_labels[i] == ORDER[i] else 0 for i in range(len(ORDER))]
    print(result)
    print("sum: ", sum(result))





eval_test()


images, labels = load_data()
images = tf.keras.applications.resnet50.preprocess_input(images)
print(len(images), len(labels))
labels = labels[:, 2]
input_shape = (images.shape[1], images.shape[2], 3)


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)


randomize = np.arange(len(images))
np.random.shuffle(randomize)
images = images[randomize]
labels = labels[randomize]


def get_model():
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')

    x = base_model.output
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                          )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu',
                          )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(6)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    return model


model = get_model()
model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

checkpoint_call = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', verbose=0,
                                                     save_best_only=True, save_weights_only=False, mode='auto',
                                                     period=1)

model.fit(x=images, y=labels, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[checkpoint_call], shuffle=True)

# model.save('model.h5')

eval_test()

# p = model.predict(images[0:10])
# new_model = keras.models.load_model('model.h5')
