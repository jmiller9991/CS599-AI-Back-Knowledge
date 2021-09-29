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

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed
import os
import pandas as pd
import math

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

def tempModifyDoc(combined_vals):
    for line in combined_vals:
        x = [18, 19]
        line = np.delete(line, x)
        print(line)

    return combined_vals

#This method sets up the content and retrieves data and strings per folder
def dataModAndGrabPerFolder(folderVal):
    #add 2D array modification
    numcol = 18
    coltemp = 8
    image_array = []

    mwmk_exists = False
    temp_mod = True

    read_MWMK = ''

    print('Starting Data Gathering...')
    for x in os.listdir(workingDir):
        #made change here, will test later => if broken just folderVal
        count = 0
        new_folder_val = folderVal if folderVal else 'GP'
        if x.startswith(new_folder_val):
            dir_string = os.path.join(workingDir, x)
            print('Looking at folder ' + dir_string)
            for files in os.listdir(dir_string):
                if files.startswith('MWMK'):
                    mwmk_exists = True
                    read_MWMK = files
                    print('MWMK file found!')
                if files.startswith('VideoFrames-'):
                    pathval = os.path.join(dir_string, files)
                    for img in os.listdir(pathval):
                        print (f'Loading Video Frame {os.path.join(pathval, img)}')
                        image_array.append(os.path.join(pathval, img))
                        count += 1
                        # images = cv2.imread(os.path.join(pathval, img))
                        # im = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                        #
                        # imageArray = np.append(imageArray, im)
                    print('Files Loaded')

            image_array = np.array(image_array)

            if mwmk_exists:
                print('Concatenating MWK and MWM')
                myfile1 = pd.read_csv(os.path.join(dir_string, read_MWMK))

                combined_vals = myfile1.to_numpy(dtype=np.int)

                print(f'b4: imageArray: {image_array.shape} array1: {combined_vals.shape}')

                # combinded_vals = np.append(combinded_vals, [array1])

                print(f'aftr: imageArray: {image_array.shape} array1: {combined_vals.shape}')

                if temp_mod:
                    combined_vals = tempModifyDoc(combined_vals)

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

                return combined_vals, image_array


def frameSort(image_array, combined_vals):
    total_key_frames = combined_vals.shape[0]
    key_inc = 3.75
    key_index_float = 0
    final_video_frames = []

    for kindex in range(total_key_frames):
        vid_index = math.floor(key_index_float)
        final_video_frames.append(image_array[vid_index])
        key_index_float += key_inc

    return final_video_frames


#This method gets the file names as images
def loadAsImg(imageArr):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(imageArr.shape[0]):
        rawImgData = tf.io.read_file(imageArr[i])
        imgData = tf.io.decode_png(rawImgData)
        conversion = tf.image.convert_image_dtype(imgData, tf.float32)
        ta = ta.write(i, conversion)
        # tf.print(conversion.shape)
    return ta.stack()

#This method manages and sets up the training model to prevent overworking the GPU
def buildTrainingModel(datastrings, inputimages):
    print('Starting to Develop the Training Model...')
    group_size = 50
    #superLists are list that divide training sets into groups of 60 (variable) frames and labels
    super_list_frame = []
    super_list_label = []

    for i in range(0, len(inputimages), group_size):
        x = inputimages[i:(i + group_size)]
        if x.shape[0] == group_size:
            super_list_frame.append(x)
            print(f'shape x: {x.shape}')

    for i in range(0, len(datastrings), group_size):
        y = datastrings[i:(i + group_size)]
        if y.shape[0] == group_size:
            super_list_label.append(y)
            print(f'shape y: {y.shape}')

    np_list_frame = np.array(super_list_frame)

    print(f'pm: {np_list_frame.shape}')

    imageset = tf.data.Dataset.from_tensor_slices(np_list_frame)
    dataset = tf.data.Dataset.from_tensor_slices(super_list_label)

    data_map = imageset.map(loadAsImg)

    data_zip = tf.data.Dataset.zip((data_map, dataset))

    print('Data Collected')

    return data_zip

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
    model.add(InputLayer(input_shape=inputShape))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=6, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(3)))
    # model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(3)))
    # model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(3)))
    model.add(TimeDistributed(Flatten()))

    print('Developing Class Counter')
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(classCnt, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    epochs = 200
    batch_size = 1

    return model, epochs, batch_size



def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    combinded_vals, image_array = dataModAndGrabPerFolder('GP2')

    final_video_frames = frameSort(image_array, combinded_vals)

    numpy_final_video_frames = np.array(final_video_frames)

    print(f'numpy_final_video_frames shape {numpy_final_video_frames.shape}')

    data_zipped = buildTrainingModel(combinded_vals, numpy_final_video_frames)

    # data_zipped = data_zipped.batch(2)
    # for thing in data_zipped:
    #     print(thing[0].numpy().shape)
    #     print(thing[1].numpy().shape)

    model, epochs, batch_size = buildModel((50, 426, 240, 3), 27)

    data_zipped = data_zipped.batch(batch_size)

    model.fit(data_zipped, epochs=epochs, batch_size=batch_size)

    print('Main')

if __name__ == '__main__':
    main()


