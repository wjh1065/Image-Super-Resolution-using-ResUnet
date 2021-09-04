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

def SRCNN():
    inputs = Input((256, 256, 3))
    x = Conv2D(64, (9, 9), padding="same", activation="relu")(inputs)
    x = Conv2D(32, (1,1), padding="same", activation="relu")(x)
    outputs = Conv2D(3, (5,5), padding="same")(x)
    model = Model(inputs, outputs)
    return model


# with strategy.scope():
model = SRCNN()

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
fig1.savefig('./results/epoch10000_SRCNN.png',dpi= 300)
model.save('./results/epoch10000_SRCNN.h5')
