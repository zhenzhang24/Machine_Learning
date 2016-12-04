import sys
import numpy as np

def conv_column_vec(arr):
    return np.array([arr]).T

def load_data(file_name):
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
        features['petal_lenght'] = float(entry[2])
        features['petal_width'] = float(entry[3])
        data.append(features)
        target.append(entry[4])
        
    return (data, target)

def calc_accuracy(pred_val, real_val):
    err = 0
    for i in range(len(real_val)):
        if real_val[i] != pred_val[i]:
            err += 1
    return (1-float(err)/len(real_val))*100
