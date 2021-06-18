################################################################################
#  Jacob Miller   ##############################################################
#  4/1/2021       ##############################################################
#  VideoManip.py  ##############################################################
################################################################################
################################################################################
# This code will do pre-processing for video data ##############################
################################################################################

import cv2
import sys
import os
import math

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

#This method will get the videos and save them as an array or arrays
def getAndModifyVideos():
    skipframe = 1
    modWidth = math.floor(1920/2.5)
    modHeight = math.floor(1080/2.5)
    pathFileName = '.png'
    for x in os.listdir(workingDir):
        if x.startswith('GP'):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(os.path.join(workingDir, x)):
                if files.endswith('.mkv'):
                    print('Working with file ' + os.path.join(dirString, files))
                    vidCap = cv2.VideoCapture(os.path.join(dirString, files))
                    success, frame = vidCap.read()
                    pathString = os.path.join(dirString, 'VideoFrames-' + str(modWidth) + '-' + str(modHeight))
                    if (not os.path.exists(os.path.join(dirString, 'VideoFrames-' + str(modWidth) + '-' + str(modHeight)))):
                        print('Making folder ' + pathString)
                        os.mkdir(pathString)
                    count = 0

                    while success:
                        print('Resizing frame ' + str(count) + ' to ' + str(modWidth) + ' ' + str(modHeight))
                        newframe = cv2.resize(frame, (modWidth, modHeight))
                        newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)
                        newFileName = 'frame' + '_' + str(modWidth) + '_' + str(modHeight) + '_' + "%05d" % count + pathFileName
                        if count % skipframe == 0:
                            print('Saving ' + os.path.join(pathString, newFileName))
                            cv2.imwrite(os.path.join(pathString, newFileName), newframe)
                        count += 1
                        success, frame = vidCap.read()
def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    getAndModifyVideos()



if __name__ == '__main__':
    main()