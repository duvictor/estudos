# # Multiclass semantic segmentation using DeepLabV3+
#
# **Author:** [Paulo Victor](http://github.com/soumik12345)<br>
# **Date created:** 2021/12/01<br>
# **Last modified:** 2021/09/1<br>
# **Description:** Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation and usinging in brain segmentation
#



import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
import os
import scipy
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output




# train_df = pd.read_csv(r'E:\\dataset\\rsna-miccai-brain-tumor-radiogenomic-classification\\train_labels.csv', dtype='str', nrows=20)
# train_df.head()
from UpdatedMeanIoU import UpdatedMeanIoU

batch_size = 4

resolution = 128
desired_depth    = 128
desired_width    = 128
desired_height   = 128
# define min max scaler
scaler = MinMaxScaler()
cluster = 1



# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,  exam_path, batch_size=32, dim=(desired_width, desired_height, desired_depth), n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = list(dataframe.MGMT_value)
        self.list_IDs = list([ f.path for f in os.scandir(exam_path) if f.is_dir() ])
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.exam_path = exam_path
        # self.df = dataframe
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        self.X_seq = np.empty(shape=self.dim)
        self.Y_seq = np.empty(shape=self.dim)
        # print("\n INICIO BATCH \n")
        # print("-" * 30)
        for ID, j in zip(list_IDs_temp, np.arange(0, self.batch_size, 1)):
            # Constructs path to the exam (to the 4 sequences)
            self.dcm_path = os.path.join(self.exam_path, ID)
            # print('\n loading: ' + self.dcm_path)
            self.read_nifti_file_x()
            self.read_nifti_file_y()
            X[j,] = self.X_seq
            y[j,] = self.Y_seq
        # print("-" * 30)
        # print("\n FIM BATCH \n")

        # for batch in range(batch_size):
            #pega o primeiro exame do paciente
            #loop do tamanho do batch, pega o primeiro exame de cada paciente do batch
            # self.plot3d(X[batch, 0])

        return X, y


    def plot3d(self, p):
        threshold = 0

        # p = exames.transpose(2, 1, 0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        plt.show()



    def read_nifti_file_y(self):
        # if is a directory
        if os.path.isdir(self.dcm_path):
            for seq, i in zip(os.listdir(self.dcm_path), np.arange(0, 5, 1)):
                if "_seg" in seq:
                    self.temp_path = os.path.join(self.dcm_path, seq)
                    self.slices = sitk.ReadImage(self.temp_path)
                    self.image = sitk.GetArrayFromImage(self.slices)

                    voxel = self.image

                    voxel = self.normalize_contrast(voxel)
                    # voxel = self.crop_voxel(voxel)
                    # voxel = self.resample3d(voxel)
                    voxel = self.resize_voxel(voxel, desired_width)
                    # self.plot3d(voxel)
                    voxel = np.stack([scaler.fit_transform(xx) for xx in voxel])


                    self.images = voxel

                    self.Y_seq[:, :, :] = self.images

    def read_nifti_file_x(self):

        # if is a directory
        if os.path.isdir(self.dcm_path):
            for seq, i in zip(os.listdir(self.dcm_path), np.arange(0, 5, 1)):
                if "flair" in seq:
                    self.temp_path = os.path.join(self.dcm_path, seq)
                    # self.nifti_list = os.listdir(self.temp_path)
                    self.slices = sitk.ReadImage(self.temp_path)
                    self.image = sitk.GetArrayFromImage(self.slices)

                    voxel = self.image

                    voxel = self.normalize_contrast(voxel)
                    # voxel = self.crop_voxel(voxel)
                    # voxel = self.resample3d(voxel)
                    voxel = self.resize_voxel(voxel, desired_width)
                    # self.plot3d(voxel)
                    voxel = np.stack([xx for xx in voxel])
                    # voxel = np.stack([scaler.fit_transform(xx) for xx in voxel])

                    self.images = voxel

                    self.X_seq[:, :, :] = self.images

    def resize_voxel(self, voxel,sz=64):
        output = np.zeros((sz, sz, sz), dtype=np.uint8)

        if np.argmax(voxel.shape) == 0:
            for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, sz)):
                output[i] = cv2.resize(voxel[int(s)], (sz, sz))
        elif np.argmax(voxel.shape) == 1:
            for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, sz)):
                output[:, i] = cv2.resize(voxel[:, int(s)], (sz, sz))
        elif np.argmax(voxel.shape) == 2:
            for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, sz)):
                output[:, :, i] = cv2.resize(voxel[:, :, int(s)], (sz, sz))

        return output

    def crop_voxel(self, voxel):
        if voxel.sum() == 0:
            return voxel
        keep = (voxel.mean(axis=(0, 1)) > 0)
        voxel = voxel[:, :, keep]
        keep = (voxel.mean(axis=(0, 2)) > 0)
        voxel = voxel[:, keep]
        keep = (voxel.mean(axis=(1, 2)) > 0)
        voxel = voxel[keep]
        return voxel


    def normalize_contrast(self, voxel):
        if voxel.sum() == 0:
            return voxel
        voxel = voxel - np.min(voxel)
        voxel = voxel / np.max(voxel)
        voxel = (voxel * 255).astype(np.uint8)
        return voxel

    def get_image_plane(self, data):
        x1, y1, _, x2, y2, _ = [round(j) for j in data.ImageOrientationPatient]
        cords = [x1, y1, x2, y2]

        if cords == [1, 0, 0, 0]:
            return 'Coronal'
        elif cords == [1, 0, 0, 1]:
            return 'Axial'
        elif cords == [0, 1, 0, 0]:
            return 'Sagittal'
        else:
            return 'Unknown'

    def standardization(self):
        self.image = np.stack([s.pixel_array for s in self.slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        self.image = self.image.astype(np.int16)

        # scalar range 0,1
        for slice_number in range(len(self.slices)):
            self.image[slice_number] = scaler.fit_transform(self.image[slice_number])
        return np.array(self.image, dtype=np.int16)


    def resample3d(self, image):
        # Determine current pixel spacing
        # Set the desired depth
        # Get current depth

        self.current_depth = image.shape[0]
        self.current_width = image.shape[1]
        self.current_height = image.shape[2]

        # Compute depth factor
        self.depth = self.current_depth / desired_depth
        self.width = self.current_width / desired_width
        self.height = self.current_height / desired_height
        self.depth_factor = 1 / self.depth
        self.width_factor = 1 / self.width
        self.height_factor = 1 / self.height

        image = scipy.ndimage.interpolation.zoom(image, (self.depth_factor, self.width_factor, self.height_factor),order=1)
        image = np.transpose(image)
        return image


# 01/12/2021
# defining a custom oxford_pets_image_segmentation model




def CustomNet(width=128, height=128, depth=128):
    ipt = keras.Input((width, height, depth, 1))

    # # Entry block
    # x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)

    # down sample
    conv1 = layers.Conv3D(16, 3, padding='same', name='conv1_bloco1_128')(ipt)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    pool1 = layers.MaxPool3D(padding="valid", data_format='channels_last', name='maxpool1_bloco1_64')(conv1)


    conv2 = layers.Conv3D(32, 3, padding='same', name='conv2_bloco2')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    pool2 = layers.MaxPool3D(padding="valid", data_format='channels_last', name='maxpool2_bloco2')(conv2)

    conv3 = layers.Conv3D(64, 3, padding='same', name='conv3_bloco3')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    pool3 = layers.MaxPool3D(padding="valid", data_format='channels_last', name='maxpool3_bloco3')(conv3)

    conv4 = layers.Conv3D(128, 3, padding='same', name='conv4_bloco4')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)
    pool4 = layers.MaxPool3D(padding="valid", data_format='channels_last', name='maxpool4_bloco4')(conv4)

    conv5 = layers.Conv3D(256, 3, padding='same', name='conv5_bloco5')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    pool5 = layers.MaxPool3D(padding="valid", data_format='channels_last', name='maxpool5_bloco5')(conv5)

    # up sample
    dconv_1 = layers.Conv3DTranspose(256, 3, padding='same', activation='relu')(pool5)
    up_1 = layers.UpSampling3D(2)(dconv_1)

    # up sample
    dconv0 = layers.Conv3DTranspose(128, 3, padding='same', activation='relu')(up_1)
    up0 = layers.UpSampling3D(2)(dconv0)

    dconv1 = layers.Conv3DTranspose(64, 3, padding='same', activation='relu')(up0)
    up1 = layers.UpSampling3D(2)(dconv1)  # 16x16 -> 32x32

    dconv2 = layers.Conv3DTranspose(32, 3, padding='same', activation='relu')(up1)
    up2 = layers.UpSampling3D(2)(dconv2)  # 32x32 -> 64x64

    # up2 = layers.add([up2, conv2])
    # aqui, gerar algo de 64 , concatenar com up2 e passar para a dconv3


    dconv3 = layers.Conv3DTranspose(16, 3, padding='same', activation='relu', name='conv3d_up_64')(up2)
    # dconv3 = layers.UpSampling3D(2)(dconv3)  # 32x32 -> 64x64

    # dconv4 = layers.Conv3DTranspose(cluster, 3, strides=2, padding='same', activation='softmax')(dconv3)
    dconv4 = layers.Conv3DTranspose(cluster, 3, strides=2, padding='same', activation='sigmoid')(dconv3)
    # dconv4 = layers.Conv3DTranspose(2, 3, strides=2, padding='same', activation='softmax')(dconv3)

    model = tf.keras.Model(inputs=ipt, outputs=dconv4)
    return model


def get_model(width=128, height=128, depth=128):
    # inputs = keras.Input(shape=img_size + (3,))
    inputs = keras.Input((width, height, depth, 1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling3D(2)(x)


        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(cluster, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model.
# whith dynamic shape
#width=desired_width, height=desired_height, depth=desired_depth
# model = get_model(width=desired_width, height=desired_height, depth=desired_depth)
model = CustomNet(width=desired_width, height=desired_height, depth=desired_depth)
model.summary()




lr   = 0.0001
sgd  = tf.keras.optimizers.SGD(learning_rate=lr)
SSC  = tf.keras.losses.SparseCategoricalCrossentropy()
CC   = tf.keras.losses.CategoricalCrossentropy()
# loss = UpdatedMeanIoU(num_classes=2)


# model.compile(optimizer=sgd, loss = SSC, metrics=['accuracy', UpdatedMeanIoU(num_classes=2)])
# model.compile(optimizer='adam', loss = SSC, metrics=['accuracy', UpdatedMeanIoU(num_classes=2)])
model.compile(optimizer='adadelta', loss='binary_crossentropy') #funcionou em 01/12/2021 começcou a aparecer algo em torno de 70 épocas
# model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])



train_path = 'D:\\dataset\\archive\\MICCAI_BraTS2020_TrainingData\\training'
test_path = 'D:\\dataset\\archive\\MICCAI_BraTS2020_TrainingData\\test'




training_generator = DataGenerator(exam_path=train_path, dim = (desired_width,desired_height,desired_depth), batch_size = batch_size)
test_generator = DataGenerator(exam_path=test_path, dim = (desired_width,desired_height,desired_depth), batch_size = batch_size)




TRAIN_LENGTH = training_generator.indexes.size
STEPS_PER_EPOCH= TRAIN_LENGTH // batch_size
epochs = 100




def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        #plotando o slice 64 do exame
        plt.imshow(display_list[i][64])
        plt.axis('off')
    plt.show()

def displayNovo(exame, predito):
    for i in range(0, 127, 20):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(exame[i])
        axarr[1].imshow(predito[i])
    plt.show()


#pegar o mesmo batch [0] de uma posicao aleaória (10), e depois disso pegar o x (0) e o y(1)
sample_image, sample_mask = training_generator.__getitem__(10)

sample_image = sample_image[0]
sample_mask = sample_mask[0]



fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)


fig, axarr = plt.subplots(16, 2,  sharex='col', sharey='row', figsize=(100,100))
fig.suptitle('Input Image and True Mask')

for i in range(0, 16, 1):
    axarr[i, 0].imshow(sample_image[i * 8])
    # axarr[i, 0].set_title('Input Image')

    axarr[i, 1].imshow(sample_mask[i * 8])
    # axarr[i, 1].set_title('True Mask')

for ax in axarr.flat:
    ax.label_outer()

plt.show()






def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, model.predict(sample_image[tf.newaxis, ...])[0]])
        # display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


model.fit_generator(generator=training_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=[DisplayCallback()],
                    verbose=1)





a = 45