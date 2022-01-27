###################################################################################################################################################################
#  Jacob Miller        ############################################################################################################################################
#  9/29/2021           ############################################################################################################################################
#  DenseOpticalFlow.py ############################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
# This code was obtained from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html ###############
# and will define the movement of objects in the game. In addition it should map the players movements in first person.                             ###############
###################################################################################################################################################################

import cv2
import imageio
import numpy as np

def DenseOpticalFlow(videoString, saveString):
    cap = cv2.VideoCapture(videoString)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    gifframes = []

    while (ret):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(saveString + 'fb.png', frame2)
            cv2.imwrite(saveString + 'hsv.png', rgb)

        if ret:
            prvs = next

        gifframes.append(rgb)

    imageio.mimsave(saveString + '.gif', gifframes)

    cap.release()
    cv2.destroyAllWindows()

def main():
    DenseOpticalFlow('C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays\\Movements\\TurnRight\\RightTurn5.mp4', 'C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays\\Movements\\TurnRight\\RightTurn5')

if __name__ == '__main__':
    main()

