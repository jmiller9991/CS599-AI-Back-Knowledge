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
    #add 2D array modification
    numcol = 18
    coltemp = 8
    image_array = np.empty((0, numcol))
    combinded_vals = np.empty((0, numcol))

    mwk_exists = False
    mwm_exists = False

    readMWK = ''
    readMWW = ''

    print('Starting Data Gathering...')
    for x in os.listdir(workingDir):
        #made change here, will test later => if broken just folderVal
        count = 0
        new_folder_val = folderVal if folderVal else 'GP'
        if x.startswith(new_folder_val):
            dir_string = os.path.join(workingDir, x)
            print('Looking at folder ' + dir_string)
            for files in os.listdir(dir_string):
                if files.startswith('MWK'):
                    mwk_exists = True
                    readMWK = files
                    print('MHK file found!')
                if files.startswith('MWM'):
                    mwm_exists = True
                    readMWW = files
                    print('MWM file found!')
                if files.startswith('VideoFrames-'):
                    pathval = os.path.join(dir_string, files)
                    for img in os.listdir(pathval):
                        print (f'Loading Video Frame ${os.path.join(pathval, img)}')
                        image_array = np.append(image_array, [count, os.path.join(pathval, img)])
                        count += 1
                        # images = cv2.imread(os.path.join(pathval, img))
                        # im = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                        #
                        # imageArray = np.append(imageArray, im)
                    print('Files Loaded')

            if mwm_exists and mwk_exists:
                print('Concatenating MWK and MWM')
                myfile1 = pd.read_csv(os.path.join(dir_string, readMWK))
                myfile2 = pd.read_csv(os.path.join(dir_string, readMWW))

                array1 = myfile1.to_numpy(dtype=np.int)
                array2 = myfile2.to_numpy(dtype=np.int)

                print(f'b4: array1: ${array1.shape}, array2: ${array2.shape}, combinedVals: ${combinded_vals.shape}')

                combinded_vals = np.append(combinded_vals, [array1, array2])

                print(f'aftr: array1: ${array1.shape}, array2: ${array2.shape}, combinedVals: ${combinded_vals.shape}')
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

                return combinded_vals, image_array



#This method manages and sets up the training model to prevent overworking the GPU = old method
# #def buildTrainingModel(dataStrings, inputImages, frameSkip):
#     print('Starting to Develop the Training Model...')
#     skipFrame = 3
#     superListFrame = []
#     superListLabel = []
#
#     if frameSkip:
#       print('Collecting images with frameskip...')
#       count = 0
#       for x in inputImages:
#           if count % skipFrame == 0:
#               print(f'Adding Frame {count}')
#               image = tf.io.read_file(x)
#               # image = cv2.imread(x)
#               # newImage = np.asarray(image)
#               newInputImages = np.append(newInputImages, [image])
#           count += 1
#
#           count2 = 0
#           for y in dataStrings:
#           if count2 % skipFrame == 0:
#               print(f'Adding Data {count2}')
#               newDataStrings = np.append(newDataStrings, [y])
#
#     else:
#       print('Collecting images without frameskip...')
#       for x in inputImages:
#           print('Adding images...')
#           image = tf.io.read_file(x)
#           newInputImages = np.append(newInputImages, [image])
#
#       for y in dataStrings:
#           print('Adding data...')
#           newDataStrings = np.append(newDataStrings, [y])

#This method manages and sets up the training model to prevent overworking the GPU
def buildTrainingModel(datastrings, inputimages):
    print('Starting to Develop the Training Model...')
    group_size = 50
    #superLists are list that divide training sets into groups of 60 (variable) frames and labels
    super_list_frame = []
    super_list_label = []

    for i in range(0, len(inputimages), group_size):
        super_list_frame.append(inputimages[i:(i + group_size)])

    for i in range(0, len(datastrings), group_size):
        super_list_label.append(datastrings[i:(i + group_size)])

    imageset = tf.data.Dataset.from_tensor_slices(super_list_frame)
    dataset = tf.data.Dataset.from_tensor_slices(super_list_label)

    data_map = tf.data.Dataset.map(imageset, dataset)

    data_zip = tf.data.Dataset.zip(data_map)

    print('Data Collected')

    return data_map, data_zip



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

    combinded_vals, image_array = dataModAndGrabPerFolder('GP1')

    _, data_zipped = buildTrainingModel(combinded_vals, image_array)

    print('Main')

if __name__ == '__main__':
    main()