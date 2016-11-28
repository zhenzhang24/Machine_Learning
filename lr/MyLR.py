import sys
import math
from sklearn.feature_extraction import DictVectorizer
import numpy as np

class MyLogisticRegression:
    def __init__(self, iter=10):
        self.iter = iter
        self.learning_rate = 0.2
        self.iter = iter
        self.stop_criteria = 0.01 # error threhold  for stopping


    def fit(self, X, Y):
        # add +1 to each row of input so that we get a constant b in x*w+b
        self.X = self.add_constant_vector(X)
        self.Y = self.make_label(Y)
        self.num_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.params = np.array([np.random.normal(0, 0.1, self.n_features+1)])
        iter = 0
        while 1:
            print "iter %d ---------" % iter
            print "params"
            print self.params
            self.Y_prob = self.calc_pred_results(self.X, self.params)
            print "probs"
            print self.Y_prob
            errors = self.calc_errors(self.Y_prob, self.Y)
            print "errors"
            print errors
            self.params = self.update_params(errors, self.params)
            total_errors = sum(errors)
            iter += 1
            if (abs(total_errors) < self.stop_criteria) or iter >= self.iter:
                break
            print '\n'
        return self.make_decision(self.Y_prob)

    def make_label(self, Y):
        return Y

    def add_constant_vector(self, X):
        shape = X.shape
        zeros = np.ndarray(shape=(shape[0], 1))
        zeros.fill(1)
        new_X = np.concatenate((X, zeros), axis=1)
        return new_X

    def calc_pred_results(self, X, params):
        pred = np.dot(X, params.T)
        print "predicted results"
        print pred
        return map(lambda x : 1/(1 + math.exp(-x)), pred)

    def calc_errors(self, pred, obsv):
        return obsv-pred

    def update_params(self, error, cur_params):
        delta = -np.dot(error, self.X) \
                * self.learning_rate / self.num_samples
        print "delta"
        print delta
        return cur_params - delta

    def make_decision(self, Y_prob):
        return map(lambda x : 1 if x >= 0.5 else  0, Y_prob)

X = np.array([[1,1], [2,2], [3,3], [4,4]], dtype=float)
Y = np.array([1, 1, 0, 0])
lr = MyLogisticRegression(iter=100)
print "FINAl Result"
print lr.fit(X, Y)
