import keras
import tensorflow as tf
from keras import layers
from IPython.display import clear_output
import matplotlib.pyplot as plt

# https://blog.keras.io/building-autoencoders-in-keras.html





input_img = keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')





from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()



x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




print(x_train.shape)
print(x_test.shape)


sample_image = x_train[0]
sample_mask = x_train[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        #plotando o slice 64 do exame
        plt.imshow(display_list[i].reshape(28,28))
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





autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                callbacks=[DisplayCallback()],
                validation_data=(x_test, x_test))

a = 45


