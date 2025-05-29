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

#workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'
workingDir = '/home/millerjs/Desktop/Gameplays/'

#This method will get the videos and save them as an array or arrays
def getAndModifyVideos():
    # variables
    win_len = 1920
    win_height = 1080

    skipframe = 1

    modWidth = math.floor(win_len/1.5)
    modHeight = math.floor(win_height/1.5)
    pathFileName = '.png'

    # search for video folders in the working directory
    for x in os.listdir(workingDir):
        # if the current video starts with GP
        if x.startswith('GP'):
            # join the workingDir string with GP string
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)

            # loop through the files and find video files ending with .mkv
            for files in os.listdir(os.path.join(workingDir, x)):
                if files.endswith('.mkv'):
                    # get the video as a OpenCV VideoCapture and read
                    print('Working with file ' + os.path.join(dirString, files))
                    vidCap = cv2.VideoCapture(os.path.join(dirString, files))
                    success, frame = vidCap.read()

                    # check if video frame folder has already been created
                    pathString = os.path.join(dirString, 'VideoFrames-' + str(modWidth) + '-' + str(modHeight))
                    if (not os.path.exists(os.path.join(dirString, 'VideoFrames-' + str(modWidth) + '-' + str(modHeight)))):
                        # if the folder is not created make it
                        print('Making folder ' + pathString)
                        os.mkdir(pathString)

                    # while the frame read is successful
                    count = 0
                    while success:
                        # resize the frame and make a new save file in the VideoFrames Folder
                        print('Resizing frame ' + str(count) + ' to ' + str(modWidth) + ' ' + str(modHeight))
                        newframe = cv2.resize(frame, (modWidth, modHeight))
                        newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)
                        newFileName = 'frame' + '_' + str(modWidth) + '_' + str(modHeight) + '_' + "%05d" % count + pathFileName

                        # if the current frame count divided by the skipframe variable has a remainder of 0, save the new frame to the folder
                        if count % skipframe == 0:
                            print('Saving ' + os.path.join(pathString, newFileName))
                            cv2.imwrite(os.path.join(pathString, newFileName), newframe)

                        # up count and read next frame
                        count += 1
                        success, frame = vidCap.read()
def main():
    global workingDir

    # search if workingDir is provided as an argument and sets the global
    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    # run the get and modify videos method
    getAndModifyVideos()



if __name__ == '__main__':
    main()