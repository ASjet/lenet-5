from cv2 import cv2
import numpy as np
import pickle
import ip
import cnn as network


img_folder_path = "img/"
img_name = "3.png"

# net = network.load()

model = network.model_folder_path+network.model_name+'.pkl'
with open(network.model_folder_path+network.model_name+'.pkl','rb') as f:
    net = pickle.load(f)

img = cv2.imread(img_folder_path+img_name)
sel = ip.cut(img,256)
dgt = ip.getDigit(sel)
flag,roi = ip.getROI(dgt)
if(flag == False):
    exit(0)
x = cv2.resize(roi, (28,28),cv2.WARP_FILL_OUTLIERS)

M = cv2.getRotationMatrix2D((14,14),30,1)
x = cv2.warpAffine(x, M, (28,28))
print(x.shape)

cv2.imshow("origin",img)
cv2.imshow("small",sel)
cv2.imshow("dgt",dgt)
cv2.imshow("resized",roi)
cv2.imshow("x",x)

nn_output = net.feedforward(x)
print(nn_output)

cv2.waitKey(0)