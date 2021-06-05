import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import uuid
import json
import os
import gzip

from nn import mnist_loader
from nn import cnn
from nn import monite
from nn import show_cnn

model_save_path = "model/"


def train(hp, model_path):
    uid = str(uuid.uuid1())
    save_path = model_path + "cnn/" + uid + '/'
    convd_size1 = np.array(hp["image_size"]) - np.array(hp["kernel_size1"]) + 1
    poold_size1 = convd_size1 // np.array(hp["pool_size"])
    convd_size2 = poold_size1 - np.array(hp["kernel_size2"]) + 1
    poold_size2 = convd_size2 // np.array(hp["pool_size"])
    flatten = np.prod(hp["feature_num"])*np.prod(poold_size2)
    FCsize = [int(flatten), 10]

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        hp["dataset_id"])
    trainset = training_data[0:1000]
    validset = validation_data[0:1000]

    net = cnn.Network(hp["feature_num"], hp["image_size"],
                      hp["kernel_size1"], hp["kernel_size2"], hp["pool_size"], FCsize)
    monitor = net.SGD(trainset, hp["epoch"], hp["mini_batch_size"], hp["eta"],
                      evaluation_data=validset, lmbda=hp["lmbda"],
                      monitor_evaluation_cost=True,
                      monitor_evaluation_accuracy=True,
                      monitor_training_cost=False,
                      monitor_training_accuracy=False)

    accuracy = monitor[1][-1]
    os.mkdir(save_path)
    with gzip.open(save_path+"model.pkl.gz", 'wb') as f:
        pickle.dump(net, f)
    with open(save_path+'monitor.pkl', 'wb') as f:
        f.write(pickle.dumps(monitor))

    info = {
        "GUID": uid,
        "Accuracy": accuracy,
        "TrainSet_Size": len(trainset),
        "Feature_Num": hp["feature_num"],
        "Kernel_Size1": hp["kernel_size1"],
        "Kernel_Size2": hp["kernel_size2"],
        "Pool_size": hp["pool_size"],
        "FCL_Size": FCsize,
        "Epoch": hp["epoch"],
        "Mini_Batch_Size": hp["mini_batch_size"],
        "Eta": hp["eta"],
        "Lambda": hp["lmbda"],
        "Mu": hp["mu"]
    }
    with open(save_path+"info.json", 'w') as f:
        json.dump(info, f)

    return uid


def showKernel(net):
    with gzip.open("../data/test.pkl.gz", 'rb') as f:
        test_set = pickle.load(f)
    for x, y in test_set:
        src = np.reshape(x, hyperparameter["image_size"])
        convd1 = net.cpl.conv(src, 1)
        poold1 = net.cpl.pooling(convd1, 1)
        convd2 = net.cpl.conv(poold1, 2)
        poold2 = net.cpl.pooling(convd2, 2)

        for i, l1 in enumerate(convd1):
            c1 = l1 if (i == 0) else np.vstack((c1, l1))

        for i, l2 in enumerate(poold1):
            p1 = l2 if (i == 0) else np.vstack((p1, l2))

        for i, l3 in enumerate(convd2):
            for j, ll3 in enumerate(l3):
                c2 = ll3 if (i == 0 and j == 0) else np.vstack((c2, ll3))

        for i, l4 in enumerate(poold2):
            for j, ll4 in enumerate(l4):
                p2 = ll4 if (i == 0 and j == 0) else np.vstack((p2, ll4))

        cv2.imshow("src", src)
        cv2.imshow("convd1", c1)
        cv2.imshow("poold1", p1)
        cv2.imshow("convd2", c2)
        cv2.imshow("poold2", p2)
        key = cv2.waitKey(0)
        if(key == 27):  # Type ESC to break
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("../model/hyperparameters_cnn.json", 'r') as f:
        hyperparameter = json.load(f)
    uid = train(hyperparameter, model_save_path)
    print(uid)
    monite.disp(model_save_path, [uid])
    # hits,all = show_cnn.testNN(test_data,save_path+"model.pkl.gz")
    # print("Accuracy on testset: %d / %d" % (hits, all))
