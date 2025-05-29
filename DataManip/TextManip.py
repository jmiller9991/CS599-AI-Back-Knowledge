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

#workingDir = 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays'
workingDir = '/home/millerjs/Desktop/Gameplays'

#This method will modify the WK_ files in each GP# folder
def modifyWMK(fileIn, fileOut):
    # Create a file reader and writer
    fileRead = open(fileIn, "r")
    fileWrite = open(fileOut, "w+")

    # loop through all lines in read file
    for line in fileRead:
        # split the line and get the last 25 data points, then only show values 1-5
        array = line.split(",")
        array = array[:-25]
        array = array[1:5]

        # a temporary array to hold data
        worksarray = []

        # look at the values in the line (now saved as variable array) and do manipulations
        for val in array:
            # removes new line character then checks if value is U or D and change it to 0 or 1 respectively
            # also values that are not digits, replace it with 1
            # lastly, put any numerical value on line
            val = val.rstrip('\n')
            if val == 'U' or val == 'None':
                worksarray.append('0')
            elif val == 'D':
                worksarray.append('1')
            elif not val.lstrip('-').isdigit():
               worksarray.append('1')
            else:
                worksarray.append(val)

        # convert array to string and write line to write file
        str = ''
        for x in worksarray:
            str += x + ','

        str = str[:-1]

        #str = str[:7]

        str += '\n'

        fileWrite.writelines(str)

    # close read and write files at end of the loop
    fileRead.close()
    fileWrite.close()

    print("Wrote New WMK_ methods under MWMK_ file labels")


#This method searches all folders in a provided working directory for folders starting in GP
#For all folders with GP as the start, it will get the files starting with WK and WM as strings and then makes an output
#string where the files will start with MWK and MWM respectively
def searchWMKFiles(starts_with):
    wmk_files_in = []
    wmk_files_out = []
    for x in os.listdir(workingDir):
        if x.startswith(starts_with):
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

    # get workingdir from arguments if available
    if len(sys.argv) > 2:
        workingDir = sys.argv[1]

    # search for file to modify and create a location string for the output
    wmk_files_in, wmk_files_out = searchWMKFiles(starts_with='GP3')

    # for all files in the list of files in
    for i in range(len(wmk_files_in)):
        # modify the file and save it to the output
        modifyWMK(wmk_files_in[i], wmk_files_out[i])

if __name__ == '__main__':
    main()