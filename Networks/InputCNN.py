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
from tensorboard.summary.v1 import image
from tensorflow.keras import models as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
#from keras.layers import TimeDistributed
from tensorflow.keras.layers import TimeDistributed
import os
import pandas as pd
import math
from datetime import datetime

from tensorflow.python.ops.signal.shape_ops import frame

# from matplotlib import pyplot as plt
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

#workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'
workingDir = '/home/millerjs/Desktop/Gameplays/'

# This is a simple helper method used during data collection, it removes the mouse data temporarily
def tempModifyDoc(combined_vals):
    for line in combined_vals:
        x = [18, 19]
        line = np.delete(line, x)
        print(line)

    return combined_vals

#This method sets up the content and retrieves data and strings per folder
def dataModAndGrabPerFolder(folderVal, width, height):
    # variables
    numcol = 18
    coltemp = 8
    image_array = []

    mwmk_exists = False
    temp_mod = False

    read_MWMK = ''

    print('Starting Data Gathering...')
    # loop through the working directory2
    for x in os.listdir(workingDir):
        count = 0
        # checks for if folderVal is provided and if not, use string GP
        new_folder_val = folderVal if folderVal else 'GP'
        if x.startswith(new_folder_val):
            # make a string of workingDir + GP labeled folders
            dir_string = os.path.join(workingDir, x)
            print('Looking at folder ' + dir_string)
            # loop through files/sub-directories in folder
            for files in os.listdir(dir_string):
                # if the file/sub-directory (will be a file) in the folder starts with 'MWMK' saves it
                # and sets if it exists to true
                if files.startswith('MWMK'):
                    mwmk_exists = True
                    read_MWMK = files
                    print('MWMK file found!')
                # if the file/sub-directory (will be a sub-directory) in the folder starts with 'VideoFrames-'
                # save each image location string in an array
                if files.startswith(f'VideoFrames-{width}-{height}'):
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

            # convert image string to numpy array
            image_array = np.array(image_array)

            # if mwmk exists
            if mwmk_exists:
                print('Concatenating MWK and MWM')
                # reads MWMK as a csv
                myfile1 = pd.read_csv(os.path.join(dir_string, read_MWMK))

                # starts combining the values by converting MWMK file as a numpy array of ints with WASD values
                combined_vals = myfile1.to_numpy(dtype=np.int_)

                print(f'b4: imageArray: {image_array.shape} array1: {combined_vals.shape}')

                # combinded_vals = np.append(combinded_vals, [array1])

                print(f'aftr: imageArray: {image_array.shape} array1: {combined_vals.shape}')

                # done for longevity/future dev but removes mouse movements
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

                # return combines WASD values and image array
                return combined_vals, image_array

# This method sorts the frames to match with combined values
def frameSort(image_array, combined_vals, key_inc=3.75):
    total_key_frames = combined_vals.shape[0]
    image_array_limit = image_array.shape[0]
    key_index_float = 0
    final_video_frames = []

    # loops through frames in array
    for kindex in range(total_key_frames):
        # get index of video
        vid_index = math.floor(key_index_float)
        # add image array at video index
        final_video_frames.append(image_array[vid_index])
        # increase key_index_float by the key skip variable key_inc
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
def buildTrainingModel(datastrings, inputimages, group_size=50):
    print('Starting to Develop the Training Model...')
    #superLists are list that divide training sets into groups of 60 (variable) frames and labels
    super_list_frame = []
    super_list_label = []

    # for input images, add to the super frame list
    for i in range(0, len(inputimages), group_size):
        x = inputimages[i:(i + group_size)]
        if x.shape[0] == group_size:
            super_list_frame.append(x)

    # for data strings, add to super label list
    for i in range(0, len(datastrings), group_size):
        y = datastrings[i:(i + group_size)]
        if y.shape[0] == group_size:
            super_list_label.append(y)

    # convert array to numpy
    np_list_frame = np.array(super_list_frame)

    print(f'pm: {np_list_frame.shape}')

    # build dataset for labels and frames
    imageset = tf.data.Dataset.from_tensor_slices(np_list_frame)
    dataset = tf.data.Dataset.from_tensor_slices(super_list_label)

    # make a map using the loadAsImg method as filter for data
    data_map = imageset.map(loadAsImg)

    # zip the data map and labels
    data_zip = tf.data.Dataset.zip((data_map, dataset))

    print('Data Collected')

    return data_zip

# I separated this out so that I can mess with this without breaking my other frame sort
# Does a similar job to frameSort
def frameSortTesting(image_array, combined_vals, key_inc=3.75):
    total_key_frames = combined_vals.shape[0]
    image_array_limit = image_array.shape[0]
    key_index_float = 0
    final_video_frames = []

    for kindex in range(total_key_frames):
        # makes the index a whole int
        vid_index = math.floor(key_index_float)
        # if vid_index does not go above the image array limit
        if vid_index < image_array_limit:
            # add image to array
            final_video_frames.append(image_array[vid_index])
            # increase index by key_index_float
            key_index_float += key_inc

    return final_video_frames

# This method manages and sets up the testing model to prevent overworking the GPU
# Separated this due to differing array sizes and testing the data manipulation function separate of training
def buildTestingModel(datastrings, inputimages, group_size=50):
    print('Starting to Develop the Testing Model...')
    #superLists are list that divide training sets into groups of 60 (variable) frames and labels
    super_list_frame = []
    super_list_label = []

    # for input images, add to the super frame list
    for i in range(0, len(inputimages), group_size):
        x = inputimages[i:(i + group_size)]
        if x.shape[0] == group_size:
            super_list_frame.append(x)

    # for data strings, add to super label list
    for i in range(0, len(datastrings), group_size):
        y = datastrings[i:(i + group_size)]
        if y.shape[0] == group_size:
            super_list_label.append(y)

    # convert array to numpy
    np_list_frame = np.array(super_list_frame)

    print(f'pm: {np_list_frame.shape}')

    # build dataset for labels and frames
    imageset = tf.data.Dataset.from_tensor_slices(np_list_frame)
    dataset = tf.data.Dataset.from_tensor_slices(super_list_label)

    # make a map using the loadAsImg method as filter for data
    data_map = imageset.map(loadAsImg)

    # zip the data map and labels
    data_zip = tf.data.Dataset.zip((data_map, dataset))

    print('Data Collected')

    return data_zip


#This method builds and compiles a model
def buildModel(inputShape, classCnt, saveFile=None):
    # INPUT SHAPE
    # MUST BE (a, b, c, d)
    # Where: a is number of images entering
    #        b is length of image
    #        c is height of image
    #        d is number of channels in the image

    if saveFile is not None:
        model = models.load_model(saveFile)
    else:
        print('Creating Model...')
        model = Sequential()

        print("Input:", inputShape)

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

        print('Compiling model')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']) # does not fit/train

        epochs = 25
        batch_size = 1

    return model, epochs, batch_size


def main():
    global workingDir
    save_file = None
    short_train = True
    model_loc = None
    frame_window = 10 #50, 25, 10
    frame_height = 308 #240, 308, 432
    frame_width = 548 #426, 548, 768

    # see if a workingDir variable is provided and if so, use that instead
    if len(sys.argv) >= 2:
        workingDir = sys.argv[1]

        # if model location is provided check if it exists and set it
        if len(sys.argv) > 2:
            model_loc_check = sys.argv[2]

            if os.path.exists(model_loc_check):
                model_loc = model_loc_check
            else:
                model_loc = None

            # if the length is longer 4, its going to check the save check is true or not
            if len(sys.argv) == 4:
                save_check = sys.argv[3]

                if save_check == 0 or save_check == True or str(save_check).lower() == "true":
                    save_file = True
                elif save_check == 1 or save_check == False or str(save_check).lower() == "false":
                    save_file = False
                else:
                    print(f'You have listed a number of inputs but {sys.argv[3]} is not an acceptable input. Saving '
                          f'will be set to false.')
                    save_file = False
        elif short_train:
            model_loc = 'ModelFiles/InputCNN-2025-4-29_16_35.keras'
        else:
            model_loc = None

    #training
    # get data and modify for training data
    combinded_vals, image_array = dataModAndGrabPerFolder('GP2', frame_width, frame_height)
    # sort the frames of the training data
    final_video_frames = frameSort(image_array, combinded_vals)
    # convert the sorted frames to a numpy array
    numpy_final_video_frames = np.array(final_video_frames)

    print(f'numpy_final_video_frames shape {numpy_final_video_frames.shape}')
    # build the training model and zip it
    data_zipped = buildTrainingModel(combinded_vals, numpy_final_video_frames, group_size=frame_window)
    # get the input shape
    sample = next(iter(data_zipped))
    input_shape = sample[0].shape
    # build the model
    model, epochs, batch_size = buildModel(input_shape, 4, model_loc)
        #(50, 240, 426, 3), 4, model_loc)

    # build a string for saving the model
    curr_time = datetime.now()
    model_str = f'./ModelFiles/InputCNN-{curr_time.year}-{curr_time.month}-{curr_time.day}_{curr_time.hour}_{curr_time.minute}.keras'
    print(f'YEAR: {curr_time.year} | MONTH: {curr_time.month} | DAY: {curr_time.day} | HOUR: {curr_time.hour} | MIN: {curr_time.minute}')
    print(model_str)

    # batch the zipped data
    data_zipped = data_zipped.batch(batch_size)

    start_time = int(datetime.now().timestamp())

    # if no model location is provided, train the model
    if model_loc is None:
        model.fit(data_zipped, epochs=epochs, batch_size=batch_size)

    # evaluate the model
    res = model.evaluate(data_zipped)

    # save the model
    model.save(model_str)

    # get a training time
    end_time = int(datetime.now().timestamp())
    print(f'Train Time: {end_time - start_time}')

    # show the training data using history value and save it
#    hist_df = pd.DataFrame(history)

    str = ''
    for item in model.metrics_names:
        str += f'{item},'

    str.rstrip(",")
    str += '\n'

    for item in res:
       str += f'{item},'

    str.rstrip(",")

    print(str)

    model_train_file=f'./results/history_model_frame_{frame_width}X{frame_height}-{frame_window}-{curr_time.year}-{curr_time.month}-{curr_time.day}_{curr_time.hour}_{curr_time.minute}.csv'
    with open(model_train_file, "w") as file:
        file.write(str)

 #   with open(model_train_file, "wb") as file:
 #       hist_df.to_csv(file)

    #evaluation
    # get and modify data for testing
    new_file_combined, new_image_array = dataModAndGrabPerFolder('GP3', frame_width, frame_height)
    # sort the frames for testing
    new_final_video_frames = frameSortTesting(new_image_array, new_file_combined)
    # convert testing array to numpy
    new_numpy_final_video_frames = np.array(new_final_video_frames)

    print(f'new_numpy_final_video_frame shape {numpy_final_video_frames.shape}')
    # build the Testing data model
    new_data_zipped = buildTestingModel(new_file_combined, new_numpy_final_video_frames, group_size=frame_window)
    # zip data to batch size
    new_data_zipped = new_data_zipped.batch(batch_size)

    # evaluate model with training data
    test_start_time = int(datetime.now().timestamp())

    results = model.evaluate(new_data_zipped)
    print(results)
    # save the results and modify data
    str = ''
    for item in model.metrics_names:
        str += f'{item},'

    str.rstrip(",")
    str += '\n'

    for item in results:
       str += f'{item},'

    str.rstrip(",")

    print(str)

    model_res_loc = f'./results/model_results_{frame_width}X{frame_height}-{frame_window}-{curr_time.year}-{curr_time.month}-{curr_time.day}_{curr_time.hour}_{curr_time.minute}.txt'
    with open(model_res_loc, "a") as file:
        file.write(str)

    '''
    # Useful for individual predictions
    #x = (50, 426, 240, 3)
    for batch_data in data_zipped:
        video, ground_labels = batch_data
        print("INPUT VIDEO:", video.shape)
        print("GROUND:", ground_labels.shape)
        pred_labels = model.predict(video)
        print("PRED:", pred_labels.shape)
    '''

    test_end_time = int(datetime.now().timestamp())

    print(f'Test Time: {test_end_time - test_start_time}')


if __name__ == '__main__':
    main()


# | ||
# || |_

# Change frame window # for 25, 10, 1 etc

