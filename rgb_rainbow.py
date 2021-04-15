from cv2 import cv2
import numpy as np
rgb = np.zeros((256,256,3),dtype=np.uint8)
for i in range(256):
    rgb[i,:,[0]] = np.arange(256)
    rgb[i,:,[1]] = np.ones((1,256))*i
    rgb[i,:,[2]] = np.abs(-np.arange(255,-1,-1)/2 - i/2)
cv2.imshow("rgb",rgb)
cv2.imshow("r",cv2.split(rgb)[0])
cv2.imshow("g",cv2.split(rgb)[1])
cv2.imshow("b",cv2.split(rgb)[2])
cv2.waitKey(0)