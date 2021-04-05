#############################
#  Jacob Miller   ###########
#  4/1/2021       ###########
#  VideoManip.py  ###########
#############################
################################################################################
# This code will do pre-processing for video data ##############################
################################################################################

import cv2
import sys

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

def getAndModifyVideos():
    print('Test')

def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]



if __name__ == '__main__':
    main()