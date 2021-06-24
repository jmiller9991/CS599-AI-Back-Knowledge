##################################################################################
#  Jacob Miller   ################################################################
#  4/1/2021       ################################################################
#  InputCNN.py    ################################################################
##################################################################################
##################################################################################
# This code will analyze all frames of a video that were edited by ###############
# VideoManip.py and the control input from TextManip.py will be    ###############
# used as labels.                                                  ###############
##################################################################################

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed
from tensorflow import data
import cv2
import pandas as pd

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

#This method sets up the content and retrieves data and strings per folder
def dataModAndGrabPerFolder(folderVal):
    numcol = 17
    coltemp = 8
    imageArray = np.empty((0, numcol))
    combindedVals = np.empty((0, numcol))
    array = np.empty((0, coltemp))

    mwkExists = False
    mwmExists = False

    readMWK = ''
    readMWW = ''

    print('Starting Data Gathering...')
    for x in os.listdir(workingDir):
        #made change here, will test later => if broken just folderVal
        newFolderVal = folderVal if folderVal else 'GP'
        if x.startswith(newFolderVal):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(dirString):
                if files.startswith('MWK'):
                    mwkExists = True
                    readMWK = files
                    print('MHK file found!')
                if files.startswith('MWM'):
                    mwmExists = True
                    readMWW = files
                    print('MWM file found!')
                if files.startswith('VideoFrames-'):
                    pathval = os.path.join(dirString, files)
                    for img in os.listdir(pathval):
                        print (f'Loading Video Frame ${os.path.join(pathval, img)}')
                        imageArray = np.concatenate([os.path.join(pathval, img)])
                        # images = cv2.imread(os.path.join(pathval, img))
                        # im = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                        #
                        # imageArray = np.append(imageArray, im)
                    print('Files Loaded')

            if mwmExists and mwkExists:
                print('Concatenating MWK and MWM')
                myfile1 = pd.read_csv(os.path.join(dirString, readMWK))
                myfile2 = pd.read_csv(os.path.join(dirString, readMWW))

                array1 = myfile1.to_numpy(dtype=np.int)
                array2 = myfile2.to_numpy(dtype=np.int)

                combindedVals = np.concatenate([array1, array2])

                print(f'array1: ${array1.shape}, array2: ${array2.shape}, combinedVals: ${combindedVals.shape}')
                print('Files Concatenated')

                # fileMWKRead = open(os.path.join(dirString, readMWK), "r")
                # fileMWMRead = open(os.path.join(dirString, readMWW), "r")
                #
                # for lines in fileMWKRead:
                #     temparray = [lines.split(',')]
                #     array = np.append(array, temparray, axis=0)
                #
                # for lines in fileMWMRead:
                #     temparray = [lines.split(',')]
                #     array = np.append(array, temparray, axis=0)
                #
                # combindedVals = np.append(combindedVals, array, axis=0)

                return combindedVals, imageArray



#This method manages and sets up the training model to prevent overworking the GPU
def buildTrainingModel(dataStrings, inputImages, frameSkip):
    print('Starting to Develop the Training Model...')
    skipFrame = 5
    newInputImages = np.empty(inputImages.shape)
    newDataStrings = np.empty(dataStrings.shape)

    if frameSkip:
        print('Collecting images with frameskip...')
        count = 0
        for x in inputImages:
            if count % skipFrame == 0:
                print(f'Adding Frame {count}')
                image = tf.io.read_file(x)
                newInputImages = np.concatenate([image])
            count += 1

        count2 = 0
        for y in dataStrings:
            if count2 % skipFrame == 0:
                print(f'Adding Data {count2}')
                newDataStrings = np.concatenate([y])

    else:
        print('Collecting images without frameskip...')
        for x in inputImages:
            print('Adding images...')
            image = tf.io.read_file(x)
            newInputImages = np.concatenate([image])

        for y in dataStrings:
            print('Adding data...')
            newDataStrings = np.concatenate([y])

    dataVal = data.Dataset.from_tensor_slices((newInputImages, newDataStrings))

    print('Data Collected')

    return dataVal


#This method builds and compiles a model
def buildModel(inputShape, classCnt):
    # INPUT SHAPE
    # MUST BE (a, b, c, d)
    # Where: a is number of images entering
    #        b is length of image
    #        c is height of image
    #        d is number of channels in the image

    print('Creating Model...')
    model = Sequential()

    print('Developing CNN...')
    model.add(Input(shape=inputShape))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Flatten()))

    print('Developing Class Counter')
    model.add(LSTM(128, return_state=True))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(classCnt, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy', 'loss'])

    epochs = 200
    batch_size = 200

    return model, epochs, batch_size



def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    combindedVals, imageArray = dataModAndGrabPerFolder('GP1')

    train_temp = buildTrainingModel(combindedVals, imageArray, True)


    print('Main')

if __name__ == '__main__':
    main()