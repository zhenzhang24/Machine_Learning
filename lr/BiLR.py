import sys
import math
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import lr
import lr_utils as utils

class BinominalLogisticRegression(lr.MyLogisticRegression):
    def __init__(self, name='bi_class_neuron', iter=10):
        lr.MyLogisticRegression(name, iter)
        self.learning_rate = 0.5
        self.accu_errors = 0

    def set_shape(self, shape):
        self.num_samples = shape[0]
        self.n_features = shape[1]

    def init_params(self):
        self.params = utils.conv_column_vec(
                np.random.normal(0, 0.1, self.n_features+1))

    def forward(self, X):
        self.X = self.add_constant_vector(X)
        self.calc_prob()
        return self.Y_prob

    def calc_derivative(self, arr):
        return map(lambda x: x*(1-x), arr)

    def calc_node_delta(self):
        symetric_error = np.apply_along_axis(
            self.calc_derivative,
            axis = 0,
            arr = self.Y_prob)
        self.errors = np.multiply(self.accu_errors, symetric_error)

    def calc_backward_error(self):
        self.back_error_distr = np.dot(self.errors, self.params.T)


    def activation_function(self, x):
        return map(lambda v : 1/(1+math.exp(-v)), x)

    def make_decision(self, Y_prob):
        return map(lambda x : 1 if x >= 0.5 else  0, Y_prob)
