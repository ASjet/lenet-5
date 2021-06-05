#! /bin/env python3
"""
* Neural Network Model Manager
* Fri Jun 04 20:38:14 CST 2021
* @Version 1.6
* @Auther ASjet
* @Email Aryan.ITFS@iCloud.com
* @License GNU GPLv3
* @Copyright © 2021, ASjet
"""

version = "1.6"

""" Changelog

@Version 1.6
    fix bugs
    optimize training options

@Version 1.5
    Refactor whole project structure

@Version 1.4
    Add train command

@Version 1.3
    Fix bugs
    Refactor functions

@Version 1.2
    Add help
    Refactor model path tree structure
    Rename commands

@Version 1.1
    Add mnt command
    Add tst command

@Version 1.0
    Add ls command
    Add la command
    Add sel command
    Add chm command

"""


import os
import json
import shutil
import pickle
from nn import *

model_path = "model/"
cfg_name = "cfg.dat"
hpname_cnn = "hyperparameters_cnn.json"
hpname_mlp = "hyperparameters_mlp.json"
model_types = [
    "mlp",
    "cnn"
]


class Model(object):
    def __init__(self, model_type, model_name, empty=False):
        self.type = ""
        self.name = ""
        self.info = {}
        self.models = []
        if(not empty):
            self.chtype(model_type)
            self.chname(model_name)

    def getModels(self):
        self.models = []
        with os.scandir(model_path + self.type) as model_folder:
            for entry in model_folder:
                if(os.path.exists(entry.path + "/info.json")):
                    self.models.append(entry.name)

    def getInfo(self, model_name):
        info = {}
        with open(model_path + self.type + '/' + model_name + "/info.json") as f:
            info = json.load(f)
        return info

    def chtype(self, model_type):
        if(model_type in model_types):
            self.type = model_type
            self.name = ""
            self.info = {}
            self.getModels()
            print("Model type changed to %s." % self.type)
            save(self)
            return True
        else:
            return False

    def chname(self, id):
        self.name = self.models[id]
        self.info = self.getInfo(self.name)
        print("Model changed to %s." % self.name)
        save(self)

    def rm(self, model_name):
        shutil.rmtree(model_path+self.type+'/'+model_name)
        print("Removed %s model %s." % (self.type, model_name))
        if(model_name == self.name):
            self.name = ""
            self.info = {}
            print("Currently using model had been removed.")
            if(self.chkEmpty()):
                self.printAll()
                new_id = int(input("Please choose a new one:"))
                if(self.chkID(new_id)):
                    self.chname(new_id)
        save(self)

    def printInfo(self):
        if(self.chkEmpty()):
            for key, value in self.info.items():
                print(key, value, sep=': ')

    def printAll(self):
        self.getModels()
        if(self.chkEmpty()):
            print("  ID|Accuracy|Layers|TrainSize|MBS|Epoch|     Eta|Lambda| Mu")
            for i, model in enumerate(self.models):
                info = self.getInfo(model)
                fmtstring = ('>' if model == self.name else ' ') + \
                    "%3d|%8d|%s|%9d|%3d|%5d|%.6f|%.4f|%.1f"
                print(fmtstring % (i, info["Accuracy"], info["Feature_Num"], info["TrainSet_Size"],
                                info["Mini_Batch_Size"], info["Epoch"], info["Eta"], info["Lambda"], info["Mu"]))
            print("Count: %d" % len(self.models))
            return True
        return False

    def chmodel(self):
        model_type = input("Please select model type[mlp/cnn]:")
        if(self.chtype(model_type)):
            if(self.printAll()):
                id = int(input("Choose one model[ID]:"))
                if(self.chkID(id)):
                    self.chname(id)
        else:
            print('Please try again by typing "chm".')

    def chkID(self, id):
        if(id < 0 or id >= len(self.models)):
            print('No such model ID "%d".' % id)
            return False
        else:
            return True

    def chkEmpty(self):
        self.getModels()
        if(len(self.models) == 0):
            print("No model found.")
            self.name = ""
            self.info = {}
            flag = input("Start by training?(y/N):")
            if(flag == 'y'):
                train(self.type)
                self.getModels()
                return True
            return False
        return True


def train(model_type):
    if(model_type == "cnn"):
        with open(model_path+hpname_cnn, 'r') as f:
            hp = json.load(f)
        # Dataset
        print("Set hyperparameters to begin")
        print("[0]Origin MNIST dataset.")
        print("[1]Rotation-Expanded MNIST dataset.")
        set_id = input("(1/9)Choose dataset[%d]:" % hp["dataset_id"])
        if(set_id != ''):
            hp["dataset_id"] = int(set_id)
        # Mini-Batch size
        mbs = input("(2/9)Mini-Batch size[%d]:" % hp["mini_batch_size"])
        if(mbs != ''):
            hp["mini_batch_size"] = int(mbs)
        # Feature num
        fn = input("(3/9)Feature Num%s:" % str(hp["feature_num"]))
        if(fn != ''):
            hp["feature_num"] = [int(f) for f in fn.split(' ')]
        # Kernel1 size
        ks1 = input("(4/9)First conv-layer kernel size%s:" %
                    str(hp["kernel_size1"]))
        if(ks1 != ''):
            hp["kernel_size1"] = [int(k) for k in ks1.split(' ')]
        # Kernel2 size
        ks2 = input("(5/9)Second conv-layer kernel size%s:" %
                    str(hp["kernel_size2"]))
        if(ks2 != ''):
            hp["kernel_size2"] = [int(k) for k in ks2.split(' ')]
        # Epoch
        epoch = input("(6/9)Epoch[%d]:" % hp["epoch"])
        if(epoch != ''):
            hp["epoch"] = int(epoch)
        # Eta
        eta = input("(7/9)Eta[%f]:" % hp["eta"])
        if(eta != ''):
            hp["eta"] = float(eta)
        # Mu
        mu = input("(8/9)Mu[%f]:" % hp["mu"])
        if(mu != ''):
            hp["mu"] = float(mu)
        # Lambda
        lmbda = input("(9/9)Lambda[%f]:" % hp["lmbda"])
        if(lmbda != ''):
            hp["lmbda"] = float(lmbda)
        uid = train_cnn.train(hp, model_path)
        info = model.getInfo(uid)
        for key, value in info.items():
            print(key, value, sep=':')

def printHelp():
    print("\nUsage:")
    print("  command [options]")
    print("\nCommands:")
    print("  train\t\tTrain new model.")
    print("  ls\t\tList information of currently selected model.")
    print("  la\t\tList informations of all model.")
    print("  rm [ID]\tRemove specified model. Currently using model will be removed when ignoring parameter.")
    print("  sel [ID]\tSelect model to use by specifing model ID. Get all model information by ignoring parameter.")
    print("  tst [ID]\tTest specified model by TestSet. Using currently using model when ignoring parameter.")
    print("  mnt [ID]\tShow monitor of specified model. Show monitor of currently using model when ignoring parameter.")
    print("  chm\t\tChange model type.")
    print("  type\t\tShow current model type.")
    print("  exit\t\tExit manager and save data.")
    print("  help\t\tShow help for commands.")
    print("  version\tShow version.")


def printHello():
    print("Neural Network Model Manager %s" % version)
    print('Type "help" for more information.')


def save(model):
    model.getModels()
    with open(model_path+cfg_name, 'wb') as f:
        f.write(pickle.dumps(model))
    with open(model_path+"index", 'w') as f:
        print(model.type, model.name, sep='\n', end='', file=f)


if __name__ == "__main__":
    printHello()
    while(True):
        if(os.path.exists(model_path+cfg_name)):
            with open(model_path+cfg_name, 'rb') as f:
                model = pickle.load(f)
            while(True):
                cl = input("$ ")
                if(cl):
                    cmd = cl.split(' ')
                    if(cmd[0] == "la"):
                        model.printAll()

                    elif(cmd[0] == "ls"):
                        model.printInfo()

                    elif(cmd[0] == "sel"):
                        if(len(cmd) == 1):
                            model.printAll()
                            id = int(input("Choose one model[ID]:"))
                        else:
                            id = int(cmd[1])
                        if(model.chkID(id)):
                            model.chname(id)

                    elif(cmd[0] == "mnt"):
                        if(len(cmd) == 1):
                            sel = model.name
                        else:
                            id = int(cmd[1])
                            if(model.chkID(id)):
                                sel = model.models[id]
                            else:
                                continue
                        print(sel)
                        path = model_path + model.type + '/'
                        monite.disp(path, sel)

                    elif(cmd[0] == "train"):
                        train(model.type)
                        save(model)

                    elif(cmd[0] == "rm"):
                        if(len(cmd) == 1):
                            sel = model.name
                        else:
                            id = int(cmd[1])
                            if(model.chkID(id)):
                                sel = model.models[id]
                            else:
                                continue
                        model.rm(sel)

                    elif(cmd[0] == "tst"):
                        if(len(cmd) == 1):
                            sel = model.name
                        else:
                            id = int(cmd[1])
                            if(model.chkID(id)):
                                sel = model.models[id]
                            else:
                                continue
                        print(sel)
                        path = model_path + model.type + '/' + sel + "/model.pkl.gz"
                        hits, all = show_cnn.testNN(show_cnn.test, path)
                        print("Accuracy on testset: %d / %d" % (hits, all))

                    elif(cmd[0] == "chm"):
                        model.chmodel()

                    elif(cmd[0] == "exit"):
                        save(model)
                        exit(0)

                    elif(cmd[0] == "type"):
                        print(model.type.upper())

                    elif(cmd[0] == "help"):
                        printHelp()

                    elif(cmd[0] == "version"):
                        printHello()

                    else:
                        print("Invalid input '%s' !" % cl)
                        printHelp()

        else:
            print("Configuration not found!")
            model = Model("", "", True)
            model.chmodel()
            save(model)
