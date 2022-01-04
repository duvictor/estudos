import keras
import tensorflow as tf
from keras import layers
from IPython.display import clear_output
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/vision/deeplabv3_plus/


IMAGE_SIZE = 128
BATCH_SIZE = 4

DATA_DIR = r"E:\PycharmProjects\brainSegmentation\dataset"
NUM_TRAIN_IMAGES = 50
NUM_VAL_IMAGES = 20

train_images = sorted(glob(os.path.join(DATA_DIR, "melanoma\*")))[:NUM_TRAIN_IMAGES]


val_images = sorted(glob(os.path.join(DATA_DIR, "melanoma\*")))[ NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES]




def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    # image = read_image(image_list)
    image = read_image(image_list, mask=True)
    mask = read_image(mask_list, mask=True)
    return image / 255, mask / 255


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# use de train images for X and the train images for Y
train_dataset = data_generator(train_images, train_images)
val_dataset = data_generator(val_images, val_images)




print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)







input_img = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

autoencoder.summary()




train_np = np.stack(list(train_dataset))



sample_image = train_np[0,0,2,:,:,:]
sample_mask = train_np[0,0,2,:,:,:]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        #plotando o slice 64 do exame
        plt.imshow(display_list[i].reshape(IMAGE_SIZE,IMAGE_SIZE))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = autoencoder.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        # display([sample_image, sample_mask, create_mask(autoencoder.predict(sample_image[tf.newaxis, ...]))])
        display([sample_image, sample_mask, autoencoder.predict(sample_image[tf.newaxis, ...])])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))





history = autoencoder.fit(train_dataset,
                epochs=100,
                shuffle=True,
                callbacks=[DisplayCallback()],
                validation_data=val_dataset)

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

a = 45
b = 90


plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()