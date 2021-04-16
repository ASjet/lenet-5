from cv2 import cv2
import numpy as np

def resize(img,length):
    h,w,rgb = np.array(img).shape
    y = (h - length) // 2
    x = (w - length) // 2
    return img[y:y+length,x:x+length]


def getDigit(img):
    gray = cv2.cvtColor(255-img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 230, 250, (3,3))
    blur = cv2.blur(edge,(11,11))
    bf = cv2.boxFilter(blur, -1, (9,9),0)
    [th_ret, bin] = cv2.threshold(bf,30,255,cv2.THRESH_BINARY)
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    res = cv2.erode(bin,ero_kernel)
    return res


def bold(img):
    bf = cv2.boxFilter(img,-1, (11,11), 0)
    [th_ret, bin] = cv2.threshold(bf, 32, 255, cv2.THRESH_BINARY)
    return bin


def getROI(img):
    x,y,w,h = cv2.boundingRect(img)
    mid_x = x + (w//2)
    mid_y = y + (h//2)
    length = (max(w,h) // 8) * 5
    return img[mid_y-length:mid_y+length,mid_x-length:mid_x+length]