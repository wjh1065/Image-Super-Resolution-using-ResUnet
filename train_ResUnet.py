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

def SR_lcs_4floor_Resunet2D(filters=8):
    inputs = Input((256, 256, 3))

    conv = Conv2D(filters * 2, kernel_size=(3, 3), padding='same', strides=(1, 1))(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters * 4, kernel_size=(3, 3), padding='same', strides=(1, 1))(conv)
    shortcut = Conv2D(filters * 4, kernel_size=(1, 1), padding='same', strides=(1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)
    output1 = add([shortcut, conv])

    res1 = BatchNormalization()(output1)
    res1 = Activation("relu")(res1)
    res1 = Conv2D(filters * 4, kernel_size=(3, 3), padding='same', strides=(2, 2))(res1)
    res1 = BatchNormalization()(res1)
    res1 = Activation("relu")(res1)
    res1 = Conv2D(filters * 8, kernel_size=(3, 3), padding='same', strides=(1, 1))(res1)
    shortcut1 = Conv2D(filters * 8, kernel_size=(1, 1), padding='same', strides=(2, 2))(output1)
    shortcut1 = BatchNormalization()(shortcut1)
    output2 = add([shortcut1, res1])

    res2 = BatchNormalization()(output2)
    res2 = Activation("relu")(res2)
    res2 = Conv2D(filters * 8, kernel_size=(3, 3), padding='same', strides=(2, 2))(res2)
    res2 = BatchNormalization()(res2)
    res2 = Activation("relu")(res2)
    res2 = Conv2D(filters * 16, kernel_size=(3, 3), padding='same', strides=(1, 1))(res2)
    shortcut2 = Conv2D(filters * 16, kernel_size=(1, 1), padding='same', strides=(2, 2))(output2)
    shortcut2 = BatchNormalization()(shortcut2)
    output3 = add([shortcut2, res2])

    res3 = BatchNormalization()(output3)
    res3 = Activation("relu")(res3)
    res3 = Conv2D(filters * 16, kernel_size=(3, 3), padding='same', strides=(2, 2))(res3)
    res3 = BatchNormalization()(res3)
    res3 = Activation("relu")(res3)
    res3 = Conv2D(filters * 32, kernel_size=(3, 3), padding='same', strides=(1, 1))(res3)
    shortcut3 = Conv2D(filters * 32, kernel_size=(1, 1), padding='same', strides=(2, 2))(output3)
    shortcut3 = BatchNormalization()(shortcut3)
    output4 = add([shortcut3, res3])

    # bridge
    conv = BatchNormalization()(output4)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters * 32, kernel_size=(3, 3), padding='same', strides=(2, 2))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters * 64, kernel_size=(3, 3), padding='same', strides=(1, 1))(conv)
    shortcut5 = Conv2D(filters * 64, kernel_size=(1, 1), padding='same', strides=(2, 2))(output4)
    shortcut5 = BatchNormalization()(shortcut5)
    output_bd = add([shortcut5, conv])

    # decoder

    uconv2 = UpSampling2D((2, 2))(output_bd)
    uconv2 = concatenate([uconv2, output4])

    uconv22 = BatchNormalization()(uconv2)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv2D(filters * 32, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv22)
    uconv22 = BatchNormalization()(uconv22)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv2D(filters * 16, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv22)
    shortcut6 = Conv2D(filters * 16, kernel_size=(1, 1), padding='same', strides=(1, 1))(uconv2)
    shortcut6 = BatchNormalization()(shortcut6)
    output7 = add([uconv22, shortcut6])

    uconv3 = UpSampling2D((2, 2))(output7)
    uconv3 = concatenate([uconv3, output3])

    uconv33 = BatchNormalization()(uconv3)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv2D(filters * 16, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv33)
    uconv33 = BatchNormalization()(uconv33)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv2D(filters * 8, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv33)
    shortcut7 = Conv2D(filters * 8, kernel_size=(1, 1), padding='same', strides=(1, 1))(uconv3)
    shortcut7 = BatchNormalization()(shortcut7)
    output8 = add([uconv33, shortcut7])

    uconv4 = UpSampling2D((2, 2))(output8)
    uconv4 = concatenate([uconv4, output2])

    uconv44 = BatchNormalization()(uconv4)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv2D(filters * 8, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv44)
    uconv44 = BatchNormalization()(uconv44)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv2D(filters * 4, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv44)
    shortcut8 = Conv2D(filters * 4, kernel_size=(1, 1), padding='same', strides=(1, 1))(uconv4)
    shortcut8 = BatchNormalization()(shortcut8)
    output9 = add([uconv44, shortcut8])

    uconv5 = UpSampling2D((2, 2))(output9)
    uconv5 = concatenate([uconv5, output1])

    uconv55 = BatchNormalization()(uconv5)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv2D(filters * 4, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv55)
    uconv55 = BatchNormalization()(uconv55)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv2D(filters * 2, kernel_size=(3, 3), padding='same', strides=(1, 1))(uconv55)
    shortcut9 = Conv2D(filters * 2, kernel_size=(1, 1), padding='same', strides=(1, 1))(uconv5)
    shortcut9 = BatchNormalization()(shortcut9)
    output10 = add([uconv55, shortcut9])

    output_layer = Conv2D(3, (1, 1), padding="same", activation="relu")(output10)
    model = Model(inputs, output_layer)

    return model


# with strategy.scope():
model = SR_lcs_4floor_Resunet2D()

model.summary()
opt = Adam(learning_rate=1e-5)
# opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss='mse')

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
fig1.savefig('./results/epoch10000_sigmoid.png',dpi= 300)
model.save('./results/epoch10000_sigmoid.h5')