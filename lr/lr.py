import sys
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np
import lr_utils as utils

g_debug = 0

class MyLogisticRegression:
    def __init__(self, iter=10, multiclass=False):
        self.iter = iter
        self.multiclass = multiclass
        self.learning_rate = 0.2
        self.iter = iter
        self.stop_criteria = 0.01 # error threhold  for stopping
        self.vectorizer = DictVectorizer(sparse=False)

    def train(self, filename):
        (self.data, self.target) = utils.load_data(filename)
        self.X = self.vectorizer.fit_transform(self.data)
        self.Y = self.make_label(self.target)
        return self.fit(self.X, self.Y)


    def fit(self, X, Y):
        # add +1 to each row of input so that we get a constant b in x*w+b
        self.X = self.add_constant_vector(X)
        self.num_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.init_params()
        iter = 0
        while 1:
            if g_debug:
                print "iter %d ---------" % iter
                print "params"
                print self.params
            self.Y_prob = self.calc_prob(self.X, self.params)
            errors = self.calc_errors(self.Y_prob, self.Y)
            self.params = self.update_params(errors, self.params)
            total_errors = 0
            for err in errors.flat:
                total_errors += abs(err)
            iter += 1
            if g_debug:
                print "probs"
                print self.Y_prob
                print "errors"
                print total_errors
                print '\n'

            if (abs(total_errors) < self.stop_criteria) or iter >= self.iter:
                break
        return self.make_decision(self.Y_prob)

    # params is a matrix of n_features+1 * n_labels, each column 
    # corresponds to a class and has the params for predicting 
    # an instance belongs to this particular class.
    def init_params(self):
        self.params = np.random.normal(0,
                0.1, 
                size=(self.n_features+1, self.num_classes))

    def make_label(self, target):
        self.label_binarizer = preprocessing.LabelBinarizer()
        binary_labels = self.label_binarizer.fit_transform(target)
        self.num_classes = len(self.label_binarizer.classes_)
        return binary_labels

    def add_constant_vector(self, X):
        shape = X.shape
        zeros = np.ndarray(shape=(shape[0], 1))
        zeros.fill(1)
        new_X = np.concatenate((X, zeros), axis=1)
        return new_X

    # the returned results is a matrix shaped num_samples * num_labels.
    # each row is a linear dot prod of weights with features over 
    # different classes.
    def calc_prob(self, X, params):
        pred = np.dot(X, params)
        return self.apply_activation(pred)

    # let's default it to softmax
    def apply_activation(self, linear_results):
        return np.apply_along_axis(
                self.activation_function, 
                axis=1, 
                arr=linear_results)

    def activation_function(self, x):
        exp_x = map(math.exp, x)
        total = sum(exp_x)
        return map(lambda i : i/total, exp_x)

    # the returned result is a n_samples * n_labels matrix
    # each row represents errors distribution 
    # over different classes of an sample.
    def calc_errors(self, pred, obsv):
        return obsv-pred

    # error.T * X is a n_features + 1 * n_labels matrix,
    # Each column represent the gradient decendent of params over a class.
    def update_params(self, error, cur_params):
        delta = -np.dot(self.X.T, error) \
                * self.learning_rate / self.num_samples
        if g_debug:
            print "delta"
            print delta
        return cur_params - delta

    def make_decision(self, Y_prob):
        return self.label_binarizer.inverse_transform(Y_prob)
        #return np.argmax(Y_prob, axis=1)

lr = MyLogisticRegression(iter=int(sys.argv[2]))
g_debug = int(sys.argv[3])
res = lr.train(sys.argv[1])
accuracy = utils.calc_accuracy(res, lr.target)
print "FINAl RESULTS"
print res
print "accuracy: %f" % accuracy
