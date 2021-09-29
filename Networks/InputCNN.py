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
from Utils.DataUtilMethods import dataModAndGrabPerFolder, frameSort

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

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


