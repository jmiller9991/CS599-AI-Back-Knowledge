###################################################################################################################################################################
#  Jacob Miller   #################################################################################################################################################
#  9/29/2021      #################################################################################################################################################
#  OpticalFlow.py #################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
# This code was obtained from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html ###############
# and will define the movement of objects in the game. In addition it should map the players movements in first person.                             ###############
###################################################################################################################################################################

import cv2
import numpy as np

def LucasKanadeOpticalFlow(videoString):
    capture = cv2.VideoCapture(videoString)

    feature_params = dict( maxCorners=100,
                           qualityLevel=0.3,
                           minDistance=7,
                           blockSize=7)

    lk_params = dict( winSize=(1920, 1080),
                      maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = capture.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    print(p0.shape)

    mask = np.zeros_like(old_frame)

    while (True):
        ret, frame = capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    capture.release()

def main():
    LucasKanadeOpticalFlow('C:\\Users\\jdude\\Desktop\\Spring2021\\CS599\\Gameplays\\GP2-DeadCenterBasic\\2021-07-13 19-16-34.mkv')

if __name__ == '__main__':
    main()
