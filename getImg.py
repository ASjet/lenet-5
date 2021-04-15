from cv2 import cv2
import numpy as np
img = cv2.imread("img/1.png",cv2.IMREAD_REDUCED_GRAYSCALE_2)

[ret,img] = cv2.threshold(img, 128,255,cv2.THRESH_BINARY_INV)

if(img.any() != None):
    cv2.imshow("camera",img)
    cv2.waitKey(0)
    cv2.destroyWindow("camera")
else:
    print("Image not found.")