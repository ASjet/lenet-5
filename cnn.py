import json
import random
import os

import cv2 as cv2
import numpy as np
from progress.bar import IncrementalBar

# import mnist_loader
import cost
import activate
import pool

model_folder_path = "cnn_model/"

model_name = "85a4dd10-a9b4-11eb-974f-9eb6d071e4f3"


def conv2D(src, kernel, res_size, biase = 0, overlapping=True, rot=False):
    if(rot):
        conv_kernel = np.flipud(kernel)
        conv_kernel = np.fliplr(kernel)
    else:
        conv_kernel = np.array(kernel)

    kernel_height,kernel_width = conv_kernel.shape

    step_x = 1 if overlapping else kernel_width
    step_y = 1 if overlapping else kernel_height

    pad_x = (kernel_width - 1) // 2
    pad_y = (kernel_height - 1) // 2
    convd = cv2.filter2D(src, -1, conv_kernel, delta = float(biase))
    res = convd[pad_y:pad_y+res_size[0], pad_x:pad_x+res_size[1]][::step_y,::step_x]
    return res


def invPool(src, kernel_size, pooled_size, res_size):
    res = np.zeros(res_size)
    kh, kw = kernel_size
    ph, pw = pooled_size
    for i in range(ph):
        for j in range(pw):
            res[i:i+kh, j:j+kw] = src[i,j]
    return res


def padding(src, pad_size):
    origin_height, origin_width = np.array(src).shape
    pad_y, pad_x = pad_size
    res = np.zeros((origin_height + 2*pad_y, origin_width + 2*pad_x))
    res[pad_y:-pad_y,pad_x:-pad_x] = src
    return res


class ConvPoolLayer(object):
    def __init__(self, feature_num, image_size, kernel_size, pool_size, pooling_fn,activator):
        self.feature_num = feature_num
        self.image_size = np.array(image_size)
        self.kernel_size = np.array(kernel_size)
        self.convd_size = self.image_size - self.kernel_size + 1
        self.pool_size = np.array(pool_size)
        self.pks = np.prod(self.pool_size)
        # self.pooled_size = self.convd_size - pool_size + 1
        self.pooled_size = self.convd_size // self.pool_size
        self.pooling_fn = pooling_fn
        self.activator=activator
        self.pool_kernel = np.ones(self.pool_size) * (1/np.prod(self.pool_size))
        # self.weights = np.array([np.round(np.random.normal(0, 1, size=kernel_size), 4)
        #                 for i in range(feature_num)])

        # self.biases = np.array([np.round(np.random.normal(0, 1, size=1), 4)
                    #    for i in range(feature_num)])
        self.weights = np.array([np.random.normal(0, 1, size=kernel_size) for i in range(feature_num)])
        self.biases = np.array([np.random.normal(0, 1, size=1) for i in range(feature_num)])

    def __repr__(self):
        for w in self.weights:
            cv2.imshow("weight",w)
            cv2.waitKey(0)
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.biases))

    def conv(self, src):
        res = [conv2D(src, k, self.convd_size, biase=b) for k, b in zip(self.weights, self.biases)]
        return np.array(res)

    def pooling(self, conv):
        pow2 = conv**2
        res = [conv2D(p, self.pool_kernel, self.convd_size,overlapping=False) for p in pow2]
        return np.array(res)

    def feedforward(self, x):
        src = np.reshape(x, self.image_size)
        convd = self.conv(src)
        activation = self.activator.act(convd)
        poold = self.pooling(activation)
        return poold.reshape((np.prod(poold.shape), 1))

    def back_prop(self, delta_full, x):
        delta_full = delta_full.reshape((self.feature_num, self.pooled_size[1], self.pooled_size[0]))
        # pad_conv = self.kernel_size-1
        delta_w = []
        delta_b = []
        z = np.reshape(x, self.image_size)
        zs = self.conv(z)
        for fl, zl in zip(delta_full, zs):
            # delta_pl = np.zeros(self.convd_size)
            # for i in range(self.pooled_size[0]):
            #     for j in range(self.pooled_size[1]):
            #         delta_pl[i:i+self.pool_size[1],j:j+self.pool_size[0]] = fl[i,j]/self.pks
            delta_pl = invPool(fl/self.pks, self.pool_size, self.pooled_size, self.convd_size)


            # extend_pl = padding(pl, pad_conv)
            # nabla_a = conv2D(extend_pl, w,rot=True)
            nabla_a = delta_pl
            delta = self.activator.derivate(zl) * nabla_a
            nabla_w = conv2D(z, delta, self.kernel_size)
            delta_w.append(nabla_w)
            delta_b.append(np.sum(delta))

        return delta_w, delta_b

    def update(self, eta, nabla_w, nabla_b, lmbda, n, mini_batch_size):
        self.weights = [(1-eta*(lmbda/n))*w-(eta/mini_batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mini_batch_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]


class Network(object):

    def __init__(self, feature_num, image_size, kernel_size, pool_size, FCsize,
                 pooling_fn=pool.max,
                 cost=cost.CrossEntropy,
                 activator=activate.Softmax):

        self.cpl = ConvPoolLayer(
            feature_num, image_size, kernel_size, pool_size, pooling_fn=pooling_fn, activator=activate.ReLU)

        self.num_layers = len(FCsize)
        self.sizes = FCsize
        self.cost = cost
        self.activator = activator
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.wv = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, img):
        a = self.cpl.feedforward(img)
        for b, w in zip(self.biases, self.weights):
            a = self.activator.act(np.dot(w, a) + b)
        # return a
        return np.argmax(a)

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
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, mu, n, cnt):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # nabla_w = np.zeros(self.weights.shape)
        # nabla_b = np.zeros(self.biases.shape)

        cpl_nabla_w = [np.zeros(w.shape) for w in self.cpl.weights]
        cpl_nabla_b = [np.zeros(b.shape) for b in self.cpl.biases]
        # cpl_nabla_w = np.zeros(np.array(self.cpl.weights).shape)
        # cpl_nabla_b = np.zeros(np.array(self.cpl.biases).shape)

        for x, y in mini_batch:
            cpl_output = self.cpl.feedforward(x)
            delta_i, delta_nabla_b, delta_nabla_w = self.backprop(cpl_output, y)
            delta_cpl_nabla_w, delta_cpl_nabla_b = self.cpl.back_prop(delta_i, x)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # nabla_b += delta_nabla_b
            # nabla_w += delta_nabla_w

            cpl_nabla_w = [nw+dnw for nw, dnw in zip(cpl_nabla_w, delta_cpl_nabla_w)]
            cpl_nabla_b = [nb+dnb for nb, dnb in zip(cpl_nabla_b, delta_cpl_nabla_b)]
            # cpl_nabla_w += delta_cpl_nabla_w
            # cpl_nabla_b += delta_cpl_nabla_b

        self.cpl.update(eta, cpl_nabla_w, cpl_nabla_b, lmbda, n, len(mini_batch))

        self.wv = [mu*v + (eta/len(mini_batch))*nw
                   for v, nw in zip(self.wv, nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w - v
                        for w, v in zip(self.weights, self.wv)]

        # self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        # self.weights = (1 - eta*lmbda/n)*self.weights - (eta/len(mini_batch))*nabla_w
        # self.biases = (1 - eta*lmbda/n)*self.weights - (eta/len(mini_batch))*nabla_b

        return cnt + 1


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # nabla_b = np.zeros(self.biases.shape)
        # nabla_w = np.zeros(self.weights.shape)
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activator.act(z)
            activations.append(activation)

        # With CrossEntropy as cost and Sigmoid as activator
        delta = self.cost.delta(zs[-1], activations[-1], y)
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

        delta_i = np.dot(self.weights[0].transpose(), delta) * self.activator.derivate(x)
        return (delta_i, nabla_b, nabla_w)


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
        # for x, y in data:
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
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    for x, y in training_data:
        img = np.reshape(x, (28, 28))
        convd = np.array(cpl.conv(img))
        poold = cpl.pooling(convd)
        for c in convd:
            print(c.shape)
            cv2.imshow("conv", np.abs(c))
            cv2.waitKey(0)
