import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv2D, concatenate, UpSampling2D, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers
from tensorflow.keras import models

framObjTrain = {'img': [],
                'mask': []
                }


## defining data Loader function
def LoadData(frameObj=None, imgPath=None, maskPath=None, shape=256):
    imgNames = os.listdir(imgPath)
    maskNames = []


    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'

    for i in range(len(imgNames)):
        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + imgNames[i])

        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))

        frameObj['img'].append(img)
        frameObj['mask'].append(mask)

    return frameObj

framObjTrain = LoadData( framObjTrain, imgPath = './Data/LR', maskPath = './Data/HR', shape = 256)

plt.subplot(1,2,1)
plt.imshow(framObjTrain['img'][1])
plt.subplot(1,2,2)
plt.imshow(framObjTrain['mask'][1])
plt.show()

class UNET(models.Model):
    def conv(x, n_f, mp_flag=True):
        x = layers.MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.05)(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    def deconv_unet(x, e, n_f):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate(axis=3)([x, e])
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    def __init__(self, org_shape):
        original = layers.Input(shape=org_shape)

        c1 = UNET.conv(original, 16, mp_flag=False)
        c2 = UNET.conv(c1, 32)
        c3 = UNET.conv(c2, 64)

        encoded = UNET.conv(c3, 128)

        z = UNET.deconv_unet(encoded, c3, 64)
        x = UNET.deconv_unet(z, c2, 32)
        y = UNET.deconv_unet(x, c1, 16)

        decoded = layers.Conv2D(3, (1, 1), activation='relu', padding='same')(y)

        # Essential Parts
        super().__init__(original, decoded)
        opt = Adam(learning_rate=1e-5)
        self.compile(optimizer=opt, loss='mse')

# with strategy.scope():
model = UNET((256,256,3))

model.summary()

train_X = np.array(framObjTrain['img'])[10:]
test_X = np.array(framObjTrain['img'])[5:10]
train_y = np.array(framObjTrain['mask'])[10:]
test_y = np.array(framObjTrain['mask'])[5:10]
print(np.array(framObjTrain['img']).shape)
print(np.array(framObjTrain['mask']).shape)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)
history = model.fit(train_X, train_y,  epochs=10000, verbose=1, validation_data=(test_X,test_y))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len[1500:], y_vloss[1500:], marker='None', c='red',label="Validation-set Loss")
plt.plot(x_len[1500:], y_loss[1500:], marker='None', c='blue', label="Train-set Loss")
fig1 = plt.gcf()
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
fig1.savefig('./results/epoch10000_RESUnet.png',dpi= 300)
model.save('./results/epoch10000_RESUnet.h5')