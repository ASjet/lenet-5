import pickle
import json
import uuid
import os
import gzip

from nn import mnist_loader
from nn import mlp
from nn import monite
from nn import show_mlp


model_save_path = "model/"


def train(hp, model_path):
    uid = str(uuid.uuid1())
    save_path = model_path + "mlp/" + uid + '/'

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        hp["dataset_id"])
    trainset = training_data
    validset = validation_data
    net = mlp.Network(hp["layer"])
    monitor = net.SGD(trainset, hp["epoch"], hp["mini_batch_size"], hp["eta"], mu=hp["mu"],
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
        "Layer": hp["layer"],
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
    with open("model/hyperparameters_mlp.json", 'r') as f:
        hyperparameter = json.load(f)
    uid = train(hyperparameter, model_save_path)
    print(uid)
    monite.disp(model_save_path, [uid])
    # hits,all = show_mlp.testNN(test_data,uid)
    # print("Accuracy on testset: %d / %d" % (hits, all))
