import numpy as np

class ReLU(object):
    @staticmethod
    def act(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def derivate(z):
        return np.where(z > 0, 1.0, 0.0)

class Sigmoid(object):
    @staticmethod
    def act(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def derivate(z):
        return Sigmoid.act(z)*(1-Sigmoid.act(z))

class Softmax(object):
    @staticmethod
    def act(z):
        # rescale = np.max(z)
        # exp_z = np.exp(z - rescale)
        # return exp_z/np.sum(exp_z)
        # exp_z = np.nan_to_num(np.exp(z))
        exp_z = np.exp(z)
        return exp_z/np.sum(exp_z)
    def derivate(z):
        return Softmax.act(z)