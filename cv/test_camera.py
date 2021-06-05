import cv2
import numpy as np
from cv import ip
from cv import config


cap = cv2.VideoCapture(config.camera_id)
while(cap.isOpened()):
    [ret,frame] = cap.read()
    if(ret == True):
        # Processed frame
        sel = ip.cut(frame,256)
        dgt = ip.getDigit(sel)
        flag,roi = ip.getROI(dgt)
        if(flag == False):
            continue
        x_size = cv2.resize(roi, (28,28), cv2.WARP_FILL_OUTLIERS)
        output = x_size

        cv2.imshow("Camera",frame)
        cv2.imshow("Captured",dgt)
        cv2.imshow("NNinput",x_size)
        key = cv2.waitKey(1)
        if(key == 27):
            break

cap.release()