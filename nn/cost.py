import numpy as np

from nn import activate

class Quadratic(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * activate.Sigmoid.derivate(z)

class CrossEntropy(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)
        # return a

class LogLikelihood(object):
    @staticmethod
    def fn(a, y):
        i = np.argmax(y)
        res = np.zeros(a)
        res[i]  = -np.nan_to_num(np.log(a[i]))
        return res
    @staticmethod
    def delta(z, a, y):
        return (a-y)
