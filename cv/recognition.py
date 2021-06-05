import os
import pickle
import gzip
import cv2

from nn import cnn, mlp
from cv import config, ip


def loadModel():
    model_type, model_name = config.readIndex()
    with gzip.open(config.model_path+model_type+'/'+model_name+"/model.pkl.gz","rb") as f:
        net = pickle.load(f)
    return net


def workflow(img, model):
    prep = ip.process(img)
    cv2.imshow("PreProcessed",prep)
    flag, obj = ip.detect(prep)
    if(flag == True):
        cv2.imshow("NNInput",obj)
        result = model.feedforward(img)
        print('\rNum: ',result, sep='', end=' ')


def camera():
    cap = cv2.VideoCapture(config.camera_id)
    net = loadModel()

    while(cap.isOpened()):
        [ret,frame] = cap.read()
        if(ret == True):
            workflow(frame, net)
            key = cv2.waitKey(1)
            if(key == 27): # Type ESC to break
                cv2.destroyAllWindows()
                break
    cap.release()
    print()


def static(img_path):
    net = loadModel()
    with os.scandir(img_path) as imgs:
        for img in imgs:
            if(img.is_file):
                frame = cv2.imread(img)
                workflow(frame, net)
    key = cv2.waitKey(1)
    if(key == 27):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # camera()
    static(config.img_path)
