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
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras import models
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv2D
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

#
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def DeepDenoiseSR_syj():
    inputs = Input((256, 256, 3))
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D()(c3)

    c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2_2)

    m1 = Add()([c2, c2_2])
    m1 = UpSampling2D()(m1)

    c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(m1)
    c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1_2)

    m2 = Add()([c1, c1_2])

    decoded = Conv2D(3, (5, 5), activation='linear', padding='same')(m2)

    model = Model(inputs, decoded)
    return model


# with strategy.scope():
model = DeepDenoiseSR_syj()

model.summary()

opt = Adam(learning_rate=1e-5)
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
fig1.savefig('./results/epoch10000_DeepDenoiseSR.png',dpi= 300)
model.save('./results/epoch10000_DeepDenoiseSR.h5')