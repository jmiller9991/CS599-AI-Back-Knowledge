##################################################################################
#  Jacob Miller        ###########################################################
#  1/26/2022           ###########################################################
#  MovementGuesser.py  ###########################################################
##################################################################################
##################################################################################
# This code will analyze all frames of a video that were edited by   #############
# DenseOpticalFlow.py and the folder names will be used as labels.   #############
# used as labels.                                                    #############
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

def buildTrainingModel(datastrings, inputimages):
    print('dasdas')

def buildModel(inputShape, classCnt):
    print('sdasdsdasd')

def main():
    global workingDir

    print('Main')

if __name__ == '__main__':
    main()