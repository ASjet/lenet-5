from cv2 import cv2
import numpy as np
import ip

# Set VideoCaptureDeviceID here
camera_id = 0

cap = cv2.VideoCapture(camera_id)

while(cap.isOpened() == True):
    [ret,frame] = cap.read()
    if(ret == True):
        # Origin captured frame
        cv2.imshow("Camera",frame)

        # Processed frame
        output = ip.process(frame)
        cv2.imshow("Output",output)

cap.release()