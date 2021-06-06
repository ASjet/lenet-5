import pickle
import gzip
import os
import numpy as np

from nn import expand_rotation

def load_data(set_id):
    if(set_id == 1):
        dataset_path = "data/mnist_expanded_rotation.pkl.gz"
        if (not os.path.exists(dataset_path)):
            expand_rotation.expand()
        f = gzip.open(dataset_path, 'rb')
    else:
        f = gzip.open('data/mnist.pkl.gz', 'rb')

    training_data, validation_data, test_data = pickle.load(f,encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(set_id):
    tr_d, va_d, te_d = load_data(set_id)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    if(set_id == 2):
        return (training_data[0:5000], validation_data[0:1000], test_data)
    else:
        return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
