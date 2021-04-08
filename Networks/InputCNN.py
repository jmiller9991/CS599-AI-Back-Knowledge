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

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed

def trainingModel():
    print('Training model!')

def buildModel(inputShape, classCnt):
    model = Sequential()

    model.add(Input(shape=(30, 282, 230, 3)))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))

def main():
    print('Main')

if __name__ == '__main__':
    main()