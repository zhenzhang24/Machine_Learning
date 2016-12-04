import sys
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np
import BiLR as lr
import lr_utils as utils

class XOR:
    def __init__(self, iter=100):
        self.iter = iter
        self.label_binarizer = preprocessing.LabelBinarizer()
        self.neuron_and1 = lr.BinominalLogisticRegression('and1')
        self.neuron_and2 = lr.BinominalLogisticRegression('and2')
        self.neuron_or = lr.BinominalLogisticRegression('or')

    def arr_not(self, arr, col):
        arr[:,col] = 1 - arr[:,col]
        return arr
        #return map(lambda x: 1-x, arr)

    def forward(self, X):
        x_not_y = np.array(X)
        x_not_y = self.arr_not(x_not_y, 1)
        not_x_y = np.array(X)
        not_x_y = self.arr_not(not_x_y, 0)

        self.neuron_and1.forward(x_not_y)
        #self.neuron_and1.debug()
        self.neuron_and2.forward(not_x_y)
        #self.neuron_and2.debug()


        l1_out_1 = self.neuron_and1.Y_prob
        l1_out_2 = self.neuron_and2.Y_prob

        l2_in = np.concatenate((l1_out_1, l1_out_2), axis=1)
        or_output = self.neuron_or.forward(l2_in)
        #self.neuron_or.debug()


    def backward(self, Y):
        self.neuron_or.accu_errors = Y - self.neuron_or.Y_prob
        
        #self.neuron_or.debug()
        self.neuron_or.calc_node_delta()
        #print "or node delta"
        #print self.neuron_or.node_delta
        self.neuron_or.calc_backward_error()

        self.back_propagate(self.neuron_and1, 
                self.neuron_or, 0)
        self.back_propagate(self.neuron_and2, 
                self.neuron_or, 0)

        self.neuron_and1.calc_node_delta()
        self.neuron_and2.calc_node_delta()

        self.neuron_or.update_params()
        self.neuron_and1.update_params()
        self.neuron_and2.update_params()

    # feedback from neuron_out to neuron_in, where neuron_in's input
    # index to neuron_out is neuron_index.
    def back_propagate(self, neuron_in, neuron_out, neuron_index):
        neuron_in.accu_errors += \
            utils.conv_column_vec(neuron_out.back_error_distr[:,neuron_index])

    def train(self, X, Y):
        iter = 0

        self.neuron_and1.set_shape(X.shape)
        self.neuron_and2.set_shape(X.shape)
        self.neuron_or.set_shape((X.shape[0], 2))
        self.neuron_and1.init_params()
        self.neuron_and2.init_params()
        self.neuron_or.init_params()



        while 1:
            self.forward(X)
            self.backward(Y)
            #self.neuron_and1.debug()
            #self.neuron_and2.debug()
            #self.neuron_or.debug()
            iter += 1
            if iter >= self.iter:
                print self.neuron_or.Y_prob
                break

xor = XOR(iter=int(sys.argv[1]))
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.int32)
Y = np.array([[0, 1, 1, 0]]).T
xor.train(X, Y)

