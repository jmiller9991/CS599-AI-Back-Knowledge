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
import cv2

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

#This method sets up the content and retrieves data and strings per folder
def dataModAndGrabPerFolder(folderVal):
    imageArray = np.empty((0, 1))
    combindedVals = np.empty((0, 1))
    array = []

    mwkExists = False
    mwmExists = False

    readMWK = ''
    readMWW = ''

    for x in os.listdir(workingDir):
        if x.startswith(folderVal):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(dirString):
                if files.startswith('MWK'):
                    mwkExists = True
                    readMWK = files
                if files.startswith('MWM'):
                    mwmExists = True
                    readMWW = files
                if files.startswith('VideoFrames-'):
                    pathval = os.path.join(dirString, files)
                    for img in os.listdir(pathval):
                        images = cv2.imread(os.path.join(pathval, img))
                        im = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

                        np.append(imageArray, im)

            if mwmExists and mwkExists:
                fileMWKRead = open(os.path.join(dirString, readMWK), "r")
                fileMWMRead = open(os.path.join(dirString, readMWW), "r")

                for lines in fileMWKRead:
                    temparray = lines.split(',')
                    array.append(temparray)

                for lines in fileMWMRead:
                    temparray = lines.split(',')
                    array.append(temparray)

                np.append(combindedVals, array, axis=0)

                return combindedVals, imageArray



#This method manages and sets up the training model to prevent overworking the GPU
def trainingModel():
    print('Training model!')

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


    print('Main')

if __name__ == '__main__':
    main()