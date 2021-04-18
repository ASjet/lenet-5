import cv2
import network
import ip

# Set VideoCaptureDeviceID
camera_id = 0


net = network.load()
cap = cv2.VideoCapture(camera_id)

while(cap.isOpened()):
    [ret,frame] = cap.read()
    if(ret == True):
        # Processed frame
        sel = ip.cut(frame,256)
        dgt = ip.getDigit(sel)
        flag,roi = ip.getROI(dgt)
        if(flag == False):
            continue
        target = cv2.resize(roi, (28,28), cv2.WARP_FILL_OUTLIERS)
        input_layer = target.reshape(784,1)

        # Display
        # cv2.imshow("Origin",frame)
        cv2.imshow("Captured",sel)
        cv2.imshow("Filter",dgt)
        cv2.imshow("Target",target)

        print('\rNum: ',net.feedforward(input_layer),sep='',end=' ')

        key = cv2.waitKey(1)
        if(key == 27): # Type ESC to break
            break

cap.release()
print()
