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
def modifyWMK(fileIn, fileOut):
    fileRead = open(fileIn, "r")
    fileWrite = open(fileOut, "w+")

    for line in fileRead:
        array = line.split(",")
        array.pop(0)

        worksarray = []

        for val in array:
            val = val.rstrip('\n')
            if val == 'U' or val == 'None':
                worksarray.append('0')
            elif val == 'D':
                worksarray.append('1')
            elif not val.lstrip('-').isdigit():
               worksarray.append('1')
            else:
                worksarray.append(val)

        str = ''
        for x in worksarray:
            str += x + ','

        str = str[:-1]

        str += '\n'

        fileWrite.writelines(str)

    fileRead.close()
    fileWrite.close()

    print("Wrote New WMK_ methods under MWMK_ file labels")


#This method searches all folders in a provided working directory for folders starting in GP
#For all folders with GP as the start, it will get the files starting with WK and WM as strings and then makes an output
#string where the files will start with MWK and MWM respectively
def searchWMKFiles():
    wmk_files_in = []
    wmk_files_out = []
    for x in os.listdir(workingDir):
        if x.startswith('GP'):
            dirString = os.path.join(workingDir, x)
            print('Looking at folder ' + dirString)
            for files in os.listdir(os.path.join(workingDir, x)):
                if files.startswith('WMK'):
                    print('Found file ' + os.path.join(dirString, files))
                    wmk_files_in.append(os.path.join(dirString, files))
                    stringFiles = files.replace('WMK', 'MWMK')
                    wmk_files_out.append(os.path.join(dirString, stringFiles))
                    print('Will output to ' + os.path.join(dirString, stringFiles))

    return wmk_files_in, wmk_files_out

def main():
    global workingDir

    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    wmk_files_in, wmk_files_out = searchWMKFiles()

    for i in range(len(wmk_files_in)):
        modifyWMK(wmk_files_in[i], wmk_files_out[i])

if __name__ == '__main__':
    main()