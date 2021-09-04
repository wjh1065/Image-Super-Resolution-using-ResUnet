import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

model = load_model('./results/epoch10000_sigmoid.h5')

framObjTrain = {'img': [],
                'mask': []
                }


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


def predict(test_X, test_y, shape=256):

    img = test_X[:5]
    imgProc = np.array(img)
    mask = test_y[:5]

    predictions = model.predict(imgProc)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc, mask

def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(20, 10))


    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Low Res image')

    ## Adding Image sharpening step here
    ## it is a sharpening filter
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imgSharpen = cv2.filter2D(predMask, -1, filter)

    plt.subplot(1, 3, 2)
    plt.imshow(imgSharpen)
    plt.title('Predicted High Res Image')

    plt.subplot(1, 3, 3)
    plt.imshow(groundTruth)
    plt.title('actual High Res Image')
    plt.show()


framObjTrain = LoadData( framObjTrain, imgPath = './Data/LR', maskPath = './Data/HR', shape = 256)

test_X = np.array(framObjTrain['img'])[:1]
test_y = np.array(framObjTrain['mask'])[:1]
print(test_X.shape, test_y.shape)



prediction, actuals, masks = predict(test_X, test_y, model)
Plotter(actuals[0], prediction[0], masks[0])


"""
PSNR SSIM LCS version 
"""
import math
from skimage.metrics import structural_similarity as ssim
def psnr(img1, img2, max_val):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

predictions = model.predict(test_X)

for i in range(len(predictions)):

    # Input data
    test_X = np.array(framObjTrain['img'])[:1]
    sq_test_X = np.squeeze(test_X)
    input_max_val = sq_test_X.max()
    print('1', sq_test_X.shape)

    # Target data
    test_y = np.array(framObjTrain['mask'])[:1]
    sq_test_y = np.squeeze(test_y)
    target_max_val = sq_test_y.max()
    print('3', sq_test_y.shape)
    # Pred data

    pred = model.predict(test_X)
    pred_max_val = pred.max()
    final_pred = np.squeeze(pred)
    print('5', final_pred.shape)



    input_psnr = psnr(sq_test_y, sq_test_X, input_max_val)
    input_ssim = ssim(sq_test_y, sq_test_X, data_range=input_max_val, multichannel=True)

    pred_psnr = psnr(sq_test_y, final_pred, pred_max_val)
    pred_ssim = ssim(sq_test_y, final_pred, data_range=pred_max_val, multichannel=True)

    result = [round(input_psnr, 4), round(pred_psnr, 4), round(input_ssim, 4), round(pred_ssim, 4)]
    print('Result : \n', result)
