import cv2
import numpy as np

def cut(img,length):
    h,w,rgb = np.array(img).shape
    y = (h - length) // 2
    x = (w - length) // 2
    return img[y:y+length,x:x+length]

def zoom(img, rate):
    res = np.repeat(img, rate, 0).repeat(rate, 1)
    return res

def getDigit(img):
    gray = cv2.cvtColor(255-img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 230, (5,5))

    blur = cv2.blur(edge,(15,15))
    bf = cv2.boxFilter(blur, -1, (3,3),normalize=0)
    [th_ret, bin] = cv2.threshold(bf,230,255,cv2.THRESH_BINARY)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    eroded = cv2.erode(bin,erode_kernel)
    res = cv2.dilate(eroded, dilate_kernel)
    return res


def getROI(img):
    flag = True
    x,y,w,h = cv2.boundingRect(img)
    if(w <= 28 and h <= 28):
        flag = False
    mid_x = x + (w//2)
    mid_y = y + (h//2)
    length = (max(w,h) // 8) * 5
    yt = max(0,mid_y-length)
    yb = min(255,mid_y+length)
    xl = max(0, mid_x-length)
    xr = min(255, mid_x+length)
    return flag,img[yt:yb,xl:xr]

def process(img):
    sel = cut(img,256)
    dgt = getDigit(sel)
    return dgt

def detect(img):
    flag,roi = getROI(img)
    if(flag):
        return True, cv2.resize(roi, (28,28), cv2.WARP_FILL_OUTLIERS)
    else:
        return False, None