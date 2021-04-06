################################################################################
#  Jacob Miller  ###############################################################
#  4/1/2021      ###############################################################
#  TextManip.py  ###############################################################
################################################################################
################################################################################
# This code will do pre-processing for text file data ##########################
################################################################################


import os
import sys

workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'

#This method will modify the WK_ files in each GP# folder
def modifyWK(fileIn, fileOut):
    fileRead = open(fileIn, "r")
    fileWrite = open(fileOut, "w+")

    for line in fileRead:
        array = line.split(",")
        array.pop(0)

        worksarray = []

        for val in array:
            if val == 'U':
                worksarray.append('0')
            elif val == 'D':
                worksarray.append('1')

        str = ''
        for x in worksarray:
            str += x + ','

        str = str[:-1]

        str += '\n'

        fileWrite.writelines(str)

    fileRead.close()
    fileWrite.close()

    print("Wrote New WK_ methods under MWK_ file labels")

#This method will modify the WM_ files in each GP# folder
def modifyWM(fileIn, fileOut):
    fileRead = open(fileIn, "r")
    fileWrite = open(fileOut, "w+")

    for line in fileRead:
        array = line.split(",")
        array.pop(0)

        worksarray = []

        for x in array:
            if x == 'D' or x == 'None':
               worksarray.append('0')
            elif not x.isdigit() :
               worksarray.append('1')
            else:
                worksarray.append(x)

        str = ""
        for x in worksarray:
            str += x + ','

        str = str[:-1]

        str += '\n'

        fileWrite.writelines(str)

    fileRead.close()
    fileWrite.close()
    print("Wrote New WM_ methods under MWM_ file labels")

#This method searches all folders in a provided working directory for folders starting in GP
#For all folders with GP as the start, it will get the files starting with WK and WM as strings and then makes an output
#string where the files will start with MWK and MWM respectively
def searchWKWMFiles():
    wkFilesIn = []
    wkFilesOut = []
    wmFilesIn = []
    wmFilesOut = []
    for x in os.listdir(workingDir):
        if x.startswith('GP'):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(os.path.join(workingDir, x)):
                if files.startswith('WK'):
                    print('Found file ' + os.path.join(dirString, files))
                    wkFilesIn.append(os.path.join(dirString, files))
                    stringFiles = files.replace('WK', 'MWK')
                    wkFilesOut.append(os.path.join(dirString, stringFiles))
                    print('Will output to ' + os.path.join(dirString, stringFiles))
                elif files.startswith('WM'):
                    print('Found file ' + os.path.join(dirString, files))
                    wmFilesIn.append(os.path.join(dirString, files))
                    stringFiles = files.replace('WM', 'MWM')
                    wmFilesOut.append(os.path.join(dirString, stringFiles))
                    print('Will output to ' + os.path.join(dirString, stringFiles))

    return wkFilesIn, wkFilesOut, wmFilesIn, wmFilesOut

def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    wkFilesIn, wkFilesOut, wmFilesIn, wmFilesOut = searchWKWMFiles()

    for i in range(len(wkFilesIn)):
        modifyWK(wkFilesIn[i], wkFilesOut[i])

    for i in range(len(wmFilesIn)):
        modifyWM(wmFilesIn[i], wmFilesOut[i])

if __name__ == '__main__':
    main()