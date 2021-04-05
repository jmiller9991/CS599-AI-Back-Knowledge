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
import os

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

#This method will get the videos and save them as an array or arrays
def getAndModifyVideos():
    for x in os.listdir(workingDir):
        if x.startswith('GP'):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(os.path.join(workingDir, x)):
                if files.endswith('.mkv'):
                    print(files)

#Test Comment
def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    getAndModifyVideos()



if __name__ == '__main__':
    main()