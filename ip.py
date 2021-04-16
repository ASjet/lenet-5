from cv2 import cv2
import numpy as np

def cut(img,length):
    h,w,rgb = np.array(img).shape
    y = (h - length) // 2
    x = (w - length) // 2
    return img[y:y+length,x:x+length]


def getDigit(img):
    gray = cv2.cvtColor(255-img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 128, 200, (3,3))

    blur = cv2.blur(edge,(11,11))
    bf = cv2.boxFilter(blur, -1, (9,9),0)
    [th_ret, bin] = cv2.threshold(bf,30,255,cv2.THRESH_BINARY)
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    eroded = cv2.erode(bin,ero_kernel)

    bold = cv2.boxFilter(eroded,-1, (11,11), 0)
    [th_ret, res] = cv2.threshold(bold, 64, 255, cv2.THRESH_BINARY)
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
    print(yt,yb,xl,xr)
    return flag,img[yt:yb,xl:xr]


def process(img):
    sel = cut(img,256)
    dgt = getDigit(sel)
    flag,roi = getROI(dgt)
    if(flag == False):
        return False, None
    res = cv2.resize(roi, (28,28), cv2.WARP_FILL_OUTLIERS)
    return True,res