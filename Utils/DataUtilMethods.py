import os
import numpy as np
import pandas as pd
import math

from Networks.InputCNN import workingDir

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


