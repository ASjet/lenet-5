from cv2 import cv2
import numpy as np
import ip

# Set VideoCaptureDeviceID here
camera_id = 0

cap = cv2.VideoCapture(camera_id)
while(cap.isOpened()):
    [ret,frame] = cap.read()
    if(ret == True):
        # Origin captured frame
        cv2.imshow("Camera",frame)

        # Processed frame
        sel = ip.cut(frame,256)
        dgt = ip.getDigit(sel)
        bd = ip.bold(dgt)
        flag,roi = ip.getROI(bd)
        if(flag == False):
            continue
        x_size = cv2.resize(roi, (28,28), cv2.WARP_FILL_OUTLIERS)
        output = x_size

        cv2.imshow("Captured",bd)
        cv2.imshow("NNinput",x_size)
        key = cv2.waitKey(1)
        if(key == 27):
            break

cap.release()