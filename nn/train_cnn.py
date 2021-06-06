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
    trainset = training_data
    validset = validation_data

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



if __name__ == "__main__":
    with open("../model/hyperparameters_cnn.json", 'r') as f:
        hyperparameter = json.load(f)
    uid = train(hyperparameter, model_save_path)
    print(uid)
    monite.disp(model_save_path, [uid])
    # hits,all = show_cnn.testNN(test_data,save_path+"model.pkl.gz")
    # print("Accuracy on testset: %d / %d" % (hits, all))
