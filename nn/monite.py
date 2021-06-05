import pickle
import matplotlib.pyplot as plt
import numpy as np

model_path = "model/"

def disp(monitor_path, monitor_name):
    with open(monitor_path+monitor_name+'/monitor.pkl', 'rb') as f:
        monitor = pickle.load(f)
    x = np.arange(len(monitor[0]))
    plt.figure(monitor_name, figsize=(9.6, 9.6))
    plt.subplot(2, 1, 1)
    plt.plot(x, monitor[0])
    plt.title("Evaluation Set Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.subplot(2, 1, 2)
    plt.plot(x, monitor[1])
    plt.title("Evaluation Set Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.subplot(2,2,3)
    # plt.plot(x,monitor[2])
    # plt.title("Training Set Cost")
    # plt.xlabel("Epoch")
    # plt.ylabel("Cost")
    # plt.subplot(2,2,4)
    # plt.plot(x,monitor[3])
    # plt.title("Training Set Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    plt.show()


if (__name__ == "__main__"):
    with open(model_path+"index", 'r') as f:
        model_type, model_name = f.read().splitlines()
    disp(model_path+model_type+'/', model_name)
