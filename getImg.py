from cv2 import cv2
import numpy as np
import time
camera_id = 0
cap = cv2.VideoCapture(camera_id)
if(cap.isOpened() != True):
    while(cap.open(camera_id) != True):
        continue

# img = cv2.imread("img/3.png",cv2.IMREAD_REDUCED_GRAYSCALE_2)

# if(img.any() != None):
#     pass
# else:
#     exit

while(cap.isOpened()):
    [ret,frame] = cap.read()
    if(ret == True):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(frame,80,255)
        [ret,prep] = cv2.threshold(255-edge,80,255,cv2.THRESH_BINARY)
        blurd = cv2.blur(255-prep,(3,3))
        [ret,bin] = cv2.threshold(255-prep, 128,255,cv2.THRESH_BINARY)

        [ret,blurd] = cv2.threshold(blurd, 64,255,cv2.THRESH_BINARY)
        bf = cv2.boxFilter(blurd,-1,(3,3),0)
        [ret,bf] = cv2.threshold(bf, 32,256,cv2.THRESH_BINARY)

        # cv2.imshow("origin",255-prep)
        # cv2.imshow("blurd",blurd)
        cv2.imshow("bf",bf)
        key = cv2.waitKey(1)
        if (key == 27):
            break
cap.release()
# cv2.waitKey(0)
# cv2.destroyWindow("camera")