from cv2 import cv2
import numpy as np
import pickle
import ip
import network


model_folder_path = "nn_model/"
model_name = "model_acc9779.json"
img_folder_path = "img/"
img_name = "2.png"

net = network.load(model_folder_path+model_name)

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

nn_input = x.reshape(28*28,1)
nn_input[nn_input > 0] = 1


nn_output = np.argmax(net.feedforward(nn_input))

print(nn_output)

cv2.waitKey(0)