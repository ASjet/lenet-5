import cv2
import numpy as np
from cv import config

def cut(img,length):
    h,w,rgb = np.array(img).shape
    y = (h - length) // 2
    x = (w - length) // 2
    return img[y:y+length,x:x+length]

def zoom(img, rate):
    res = np.repeat(img, rate, 0).repeat(rate, 1)
    return res

def getDigit(img):
    gray = cv2.cvtColor(255 - img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    im_gray1 = np.array(gray)
    avg_gray = np.average(im_gray1)
    im_gray1 = np.where(im_gray1[:, :] < config.value, 0, 255)               #灰度图转黑白阈值
    gray3 = np.array(im_gray1, dtype='uint8')
    edge = cv2.Canny(gray, 100, 230)                                         #边缘识别阈值
    contours, hierarchy = cv2.findContours(gray3, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    whit = np.zeros_like(edge)
    whit2 = np.copy(whit)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], True)
        if (abs(area) > 300):                                                       #初步过滤小杂点
            whit2 = cv2.drawContours(whit2, contours, i, (255, 255, 255), 15)
    contours, hierarchy = cv2.findContours(whit2, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    whit3 = np.copy(whit)
    n = len(contours)
    for i in range(n):
        area = cv2.contourArea(contours[i], True)
        if (np.min(contours[i]) < 15 or np.max(contours[i] > 245)):
            continue
        if (abs(area) > 2000 and hierarchy[0][i][3]==-1):#只画出图像主体
            whit3 = cv2.drawContours(whit3, contours, i, (255, 255, 255), -1)
            j = i
            while (hierarchy[0][j][2] > -1):
                cv2.drawContours(whit3, contours, hierarchy[0][j][2],
                                 (0, 0, 0), -1)
                z = hierarchy[0][j][2]
                while (hierarchy[0][z][0] > -1):
                    cv2.drawContours(whit3, contours, hierarchy[0][z][0],
                                     (0, 0, 0), -1)
                    z = hierarchy[0][z][0]
                j = hierarchy[0][j][2]
    return whit3
    #return gray3


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
