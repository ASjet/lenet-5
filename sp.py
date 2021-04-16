from cv2 import cv2
import numpy as np
import ip

img_folder_path = "img/"
img_name = "3.png"

img = cv2.imread(img_folder_path+img_name)
sel = ip.resize(img,256)
dgt = ip.getDigit(sel)
bold = ip.bold(dgt)
roi = ip.getROI(bold)
x = cv2.resize(roi, (28,28),cv2.WARP_FILL_OUTLIERS)


# cv2.imshow("origin",img)
cv2.imshow("small",sel)
cv2.imshow("dgt",dgt)
cv2.imshow("bold",bold)
cv2.imshow("resized",roi)
cv2.imshow("x",x)

cv2.waitKey(0)