import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import pickle
import uuid
import json

import mnist_loader
import cnn
import monite_cnn
import show_cnn


model_save_path = "../model/"
monitor_save_path = "../monitor/"
uid = str(uuid.uuid1())
monitors = []


feature_num = 10
image_size = (28,28)
kernel_size = (5,5)
pool_size = (2,2)
flatten = feature_num*144
FCsize = [flatten,10]


epoch = 30
eta = 0.0001
mini_batch_size = 10
mu = 0.15
lmbda = 1


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = cnn.Network(feature_num,image_size,kernel_size,pool_size,FCsize)
monitor = net.SGD(training_data, epoch, mini_batch_size, eta,
                    evaluation_data=validation_data,lmbda=lmbda,
                    monitor_evaluation_cost=True,
                    monitor_evaluation_accuracy=True,
                    monitor_training_cost=True,
                    monitor_training_accuracy=True)

accuracy = monitor[1][-1]


with open(model_save_path+uid+".pkl", 'wb') as f:
    f.write(pickle.dumps(net))

with open(monitor_save_path+uid+'.pkl','wb') as f:
    f.write(pickle.dumps(monitor))

with open("../record","a") as f:
    f.write(uid+'\n')
monitors.append(uid)

print(uid)

info = {
    uid:
    {
        "accuracy":accuracy,
        "feature_num":feature_num,
        "kernel_size":kernel_size,
        "pool_size":pool_size,
        "FCsize:":FCsize,
        "epoch:":epoch,
        "mini_batch_size:":mini_batch_size,
        "eta:":eta,
        "lambda:":lmbda,
        "mu:":mu
    }
}
with open("../info.json",'a') as f:
    json.dump(info,f)
    f.write(",\n")


for x,y in test_data:
    convd = net.cpl.conv(np.reshape(x,(28,28)))
    pooled = net.cpl.pooling(convd)
    for i, layer in enumerate(pooled):
        if(i == 0):
            img = layer
        else:
            img = np.vstack((img,layer))
    cv2.imshow("convd",img)
    key = cv2.waitKey(0)
    if(key == 27): # Type ESC to break
        break

monite_cnn.disp(monitor_save_path,monitors)
hits,all = show_cnn.testNN(test_data,uid)
print("Accuracy on testset: %d / %d" % (hits, all))
