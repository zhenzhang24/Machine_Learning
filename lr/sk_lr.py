import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import numpy as np
import lr_utils as utils

class SkLearnLogRegression:
    def __init__(self, multi_class='multinomial'):
        self.lr = linear_model.LogisticRegression(multi_class=multi_class,
                solver='lbfgs')
        self.v = DictVectorizer()

    def load_data(self, file_name):
        f = open(file_name)
        data = []
        target = []
        for line in f.readlines()[1:]:
            entry = line.split(',')
            if len(entry) < 5:
                continue
            features = {}
            features['sepal_length'] = float(entry[0])
            features['sepal_width'] = float(entry[1])
            features['petal_length'] = float(entry[2])
            features['petal_width'] = float(entry[3])
            data.append(features)
            target.append(entry[4])
        return (data, target)

    def train(self, training_data_file):
        (self.data, self.target) = utils.load_data(training_data_file)

        self.X = self.v.fit_transform(self.data)
        self.Y = np.array(self.target)
        print self.X.shape
        return self.lr.fit(self.X, self.Y)

    def test(self, test_file):
        (test_data, test_target) = utils.load_data(test_file)
        test_result = self.predict(test_data)
        print test_result
        return (test_result, test_target)

    def predict(self, test_data):
        sample = self.v.transform(test_data)
        return self.lr.predict(sample)

training_data_file = sys.argv[1]
test_data_file = sys.argv[2]
lr = SkLearnLogRegression()
lr.train(training_data_file)
print "Config: "
print "training_data: ", training_data_file
print "testing data:", test_data_file
(predicted, test_y) = lr.test(test_data_file)
accur = utils.calc_accuracy(predicted, test_y)
print "accuracy:", accur
