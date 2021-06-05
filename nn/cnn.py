import json
import random
import os

import cv2 as cv2
import numpy as np
from numpy.core.fromnumeric import shape
from progress.bar import IncrementalBar

from nn import mnist_loader
from nn import cost
from nn import activate
from nn import pool


def conv2D(src, kernel, res_size, biase=0, overlapping=True, rot=False):
    src_y, src_x = src.shape
    res_y, res_x = res_size
    if(rot):
        conv_kernel = cv2.flip(kernel, -1)
    else:
        conv_kernel = np.array(kernel)

    convd = cv2.filter2D(src, -1, conv_kernel, delta=float(-biase))

    kernel_height, kernel_width = kernel.shape

    if(overlapping):
        pad_x = 0 if (src_x == res_x) else (kernel_width - 1) // 2
        pad_y = 0 if (src_y == res_y) else (kernel_height - 1) // 2
        res = convd[pad_y:pad_y+res_y, pad_x:pad_x+res_x]
    else:
        res_x *= kernel_width
        res_y *= kernel_height
        pad_x = (src_x - res_x) // 2
        pad_y = (src_y - res_y) // 2
        res = convd[pad_y:pad_y+res_y, pad_x:pad_x +
                    res_x][::kernel_width, ::kernel_width]
    return res


def invPool(src, kernel_size):
    kh, kw = kernel_size
    res = np.repeat(src, kh, 0).repeat(kw, 1)
    return res


def padding(src, pad_size):
    origin_height, origin_width = np.array(src).shape
    pad_y, pad_x = pad_size
    res = np.zeros((origin_height + 2*pad_y, origin_width + 2*pad_x))
    res[pad_y:-pad_y, pad_x:-pad_x] = src
    return res


class ConvPoolLayer(object):
    def __init__(self, feature_num, image_size, kernel_size1, kernel_size2, pool_size, pooling_fn, activator):
        self.feature_num = feature_num
        self.image_size = np.array(image_size)
        self.kernel_size1 = np.array(kernel_size1)
        self.kernel_size2 = np.array(kernel_size2)
        self.pool_size = np.array(pool_size)
        self.pks = np.prod(self.pool_size)
        self.convd_size1 = self.image_size - self.kernel_size1 + 1
        self.pooled_size1 = self.convd_size1 // self.pool_size
        self.convd_size2 = self.pooled_size1 - self.kernel_size2 + 1
        self.pooled_size2 = self.convd_size2 // self.pool_size
        self.pooling_fn = pooling_fn
        self.activator = activator
        self.pool_kernel = np.ones(self.pool_size) / (np.prod(self.pool_size))

        self.weights1 = np.array([np.random.normal(0, 1, size=kernel_size1)
                                  for i in range(feature_num[0])])
        self.weights2 = np.array([np.array([np.random.normal(0, 1, size=kernel_size2)
                                            for i in range(feature_num[1])])
                                  for j in range(feature_num[0])])

        # self.biases1 = np.array([np.random.normal(0, 1, size=1) for i in range(feature_num[0])])
        self.biases1 = np.ones((self.feature_num[0], 1))

        # self.biases2 = np.array([np.array([np.random.normal(0, 1, size=1) for i in range(feature_num[1])])
        self.biases2 = np.ones(self.feature_num)

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights1), repr(self.biases1))

    def conv(self, src, layout):
        if(layout == 1):
            res = [conv2D(src, k, self.convd_size1, biase=b)
                   for k, b in zip(self.weights1, self.biases1)]
        elif(layout == 2):
            res = [[conv2D(feat, w, self.convd_size2, biase=b) for w, b in zip(lw, lb)]
                   for feat, lw, lb in zip(src, self.weights2, self.biases2)]
        return np.array(res)

    def pooling(self, conv, layout):
        if(layout == 1):
            res = [conv2D(p, self.pool_kernel, self.pooled_size1,
                          overlapping=False) for p in conv]
        elif(layout == 2):
            res = [[conv2D(f, self.pool_kernel, self.pooled_size2, overlapping=False)*4 for f in feat]
                   for feat in conv]
        return np.array(res)

    def feedforward(self, x):
        src = np.reshape(x, self.image_size)
        convd = self.conv(src, 1)
        activation = self.activator.act(convd)
        poold = self.pooling(activation, 1)
        convd = self.conv(poold, 2)
        activation = self.activator.act(convd)
        poold = self.pooling(activation, 2)
        return poold.reshape((np.prod(poold.shape), 1))

    def back_prop(self, delta_full, x):
        delta_full = delta_full.reshape(
            (self.feature_num[0], self.feature_num[1], self.pooled_size2[0], self.pooled_size2[1]))
        delta_l2 = []
        delta_w1 = []
        delta_b1 = []
        delta_w2 = []
        delta_b2 = []
        z = np.reshape(x, self.image_size)
        zs = [z]
        acts = []

        z = (self.conv(z, 1))
        act = self.activator.act(z)
        acts.append(act)
        z = self.pooling(act, 1)
        zs.append(z)

        z = self.conv(zs[-1], 2)
        act = self.activator.act(z)
        acts.append(act)

        for fl1, ws, z_l1, ac in zip(delta_full/self.pks, self.weights2, zs[-1], acts[-1]):
            delta_pl = [invPool(fl2, self.pool_size) for fl2 in fl1]
            derive = self.activator.derivate(z_l1)
            extend_dpl = [padding(dpl, (self.kernel_size2-1)//2)
                          for dpl in delta_pl]
            deltas = [conv2D(edpl, w, self.pooled_size1, rot=True)*derive
                      for edpl, w in zip(extend_dpl, ws)]
            delta = np.sum(deltas, axis=0)/self.feature_num[1]
            nabla_w = [conv2D(d, a, self.kernel_size2)
                       for d, a in zip(deltas, ac)]
            nabla_b = np.sum(deltas, axis=(1, 2))
            delta_l2.append(delta)
            delta_w2.append(nabla_w)
            delta_b2.append(nabla_b)

        for dl2, w, a in zip(delta_l2/self.pks, self.weights1, acts[0]):
            delta_pl = invPool(dl2, self.pool_size)
            derive = self.activator.derivate(zs[0])
            extend_dpl = padding(delta_pl, (self.kernel_size1-1)//2)
            delta = conv2D(extend_dpl, w, self.image_size, rot=True)*derive
            nabla_w = conv2D(delta, a, self.kernel_size1)
            delta_w1.append(nabla_w)
            delta_b1.append(np.sum(delta))

        return delta_w1, delta_b1, delta_w2, delta_b2

    def update(self, eta, nabla_w1, nabla_b1, nabla_w2, nabla_b2, lmbda, n, mini_batch_size):
        self.weights1 = [(1-eta*(lmbda/n))*w - (eta/mini_batch_size)*nw
                         for w, nw in zip(self.weights1, nabla_w1)]
        self.biases1 = [b - (eta/mini_batch_size)*nb
                        for b, nb in zip(self.biases1, nabla_b1)]
        self.weights2 = [[(1-eta*(lmbda/n))*w - (eta/mini_batch_size)*nw for w, nw in zip(ws, nws)]
                         for ws, nws in zip(self.weights2, nabla_w2)]
        self.biases2 = [[b - (eta/mini_batch_size)*nb for b, nb in zip(bs, nbs)]
                        for bs, nbs in zip(self.biases2, nabla_b2)]


class Network(object):

    def __init__(self, feature_num, image_size, kernel_size1, kernel_size2, pool_size, FCsize,
                 pooling_fn=pool.max,
                 cost=cost.CrossEntropy,
                 activator=activate.Softmax):

        self.cpl = ConvPoolLayer(
            feature_num, image_size, kernel_size1, kernel_size2, pool_size, pooling_fn=pooling_fn, activator=activate.Sigmoid)

        self.num_layers = len(FCsize)
        self.sizes = FCsize
        self.cost = cost
        self.activator = activator
        # self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.biases = [np.ones((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.wv = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, img):
        a = self.cpl.feedforward(img)
        for b, w in zip(self.biases, self.weights):
            a = self.activator.act(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, mu=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        inc_cnt = 0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            cnt = 0
            for mini_batch in IncrementalBar("Epoch %d" % j, suffix='%(percent).2f%% ETA %(eta)ds').iter(mini_batches):
                # self.update_mini_batch(mini_batch, eta, lmbda, n)
                cnt = self.update_mini_batch(
                    mini_batch, eta, lmbda, mu, n, cnt)

            print("\nEpoch %d training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                if(len(evaluation_cost) > 0):
                    if(cost > evaluation_cost[-1]):
                        inc_cnt += 1
                    else:
                        inc_cnt = 0
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            if(inc_cnt >= 5):
                break
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, mu, n, cnt):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        cpl_nabla_w1 = [np.zeros(w.shape) for w in self.cpl.weights1]
        cpl_nabla_b1 = [np.zeros(b.shape) for b in self.cpl.biases1]
        cpl_nabla_w2 = [[np.zeros(w.shape) for w in ws]
                        for ws in self.cpl.weights2]
        cpl_nabla_b2 = [[np.zeros(b.shape) for b in bs]
                        for bs in self.cpl.biases2]

        for x, y in mini_batch:
            cpl_output = self.cpl.feedforward(x)
            delta_i, delta_nabla_w, delta_nabla_b = self.backprop(
                cpl_output, y)
            delta_nabla_w = delta_nabla_w
            delta_nabla_b = delta_nabla_b
            dcplnw1, dcplnb1, dcplnw2, dcplnb2 = self.cpl.back_prop(delta_i, x)

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_b += delta_nabla_b

            cpl_nabla_w1 = [nw+dnw for nw, dnw in zip(cpl_nabla_w1, dcplnw1)]
            # cpl_nabla_w2 = [nw+dnw for nw, dnw in zip(nws, dnws)
            #                 for nws, dnws in zip(cpl_nabla_w2, dcplnw2)]
            cpl_nabla_w2 = [nw+dnw for nw, dnw in zip(cpl_nabla_w2, dcplnw2)]
            # cpl_nabla_b1 = [nb+dnb for nb, dnb in zip(cpl_nabla_b1, dcplnb1)]
            # cpl_nabla_b2 = [nb+dnb for nb, dnb in zip(cpl_nabla_b2, dcplnb2)]
            cpl_nabla_b1 += dcplnb1
            cpl_nabla_b2 += dcplnb2

        self.cpl.update(eta, cpl_nabla_w1, cpl_nabla_b1,
                        cpl_nabla_w2, cpl_nabla_b2, lmbda, n, len(mini_batch))

        self.wv = [mu*v + (eta/len(mini_batch))*nw
                   for v, nw in zip(self.wv, nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w - v
                        for w, v in zip(self.weights, self.wv)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        # self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights, nabla_w)]
        # self.biases -= (eta/len(mini_batch))*nabla_b

        # self.weights = (1 - eta*lmbda/n)*self.weights - (eta/len(mini_batch))*nabla_w
        # self.biases = (1 - eta*lmbda/n)*self.weights - (eta/len(mini_batch))*nabla_b

        return cnt + 1

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # activation = self.activator.act(x)
        activation = x
        activations = [activation]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activator.act(z)
            activations.append(activation)

        # With CrossEntropy as cost and Softmax as activator
        delta = self.cost.delta(zs[-1], activations[-1], y) - y

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activator.derivate(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        delta_i = np.dot(self.weights[0].transpose(),
                         delta) * self.activator.derivate(x)
        return (delta_i, nabla_w, nabla_b)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if(__name__ == "__main__"):
    cpl = ConvPoolLayer(5, (28, 28), (5, 5), (2, 2))
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        0)
    for x, y in training_data:
        img = np.reshape(x, (28, 28))
        convd = np.array(cpl.conv(img))
        poold = cpl.pooling(convd)
        for c in convd:
            print(c.shape)
            cv2.imshow("conv", np.abs(c))
            cv2.waitKey(0)
