import sys
import tensorflow as tf
import numpy as np
import cPickle

def read_data_sets(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    data, labels = dict['data'], dict['labels']
    num_images = data.shape[0]
    depth = 3
    height = 32
    width = 32
    res_images = np.empty(shape=(num_images, depth, height, width))
            #dtype=float)
    for i in range(num_images):
        res_images[i] = np.reshape(data[i], [depth, height, width])
    res_images =np.transpose(res_images, [0, 2, 3, 1])
    del data
    return res_images, labels
