from cv2 import cv2
import numpy as np
import pickle
import ip
import mlp as network


img_folder_path = "img/"
img_name = "1.png"

net = network.load()

img = cv2.imread(img_folder_path+img_name)
sel = ip.cut(img,256)
dgt = ip.getDigit(sel)
flag,roi = ip.getROI(dgt)
if(flag == False):
    exit(0)
x = cv2.resize(roi, (28,28),cv2.WARP_FILL_OUTLIERS)


cv2.imshow("origin",img)
cv2.imshow("small",sel)
cv2.imshow("dgt",dgt)
cv2.imshow("resized",roi)
cv2.imshow("x",x)

nn_output = net.feedforward(x.reshape(28*28,1))
print(nn_output)

cv2.waitKey(0)