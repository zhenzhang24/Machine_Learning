import sys
import numpy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer

def load_data(data_file):
    f = open(data_file)
    data = []
    target = []
    uids = []
    mids = []
    for line in f.readlines():
        entry = line.split('\t')
        uid = entry[0]
        mid = entry[1]
        rating = int(entry[2])
        timestamp = int(entry[3])
        data.append({'uid':uid, 'movieid':mid})
        target.append(rating)
        uids.append(uid)
        mids.append(mid)
    f.close()
    return (data, target, uids, mids)

def train(training_data_file):
    (training_data, target, uids, mids) = load_data(training_data_file)
    v = DictVectorizer()
    M = v.fit_transform(training_data)
    print "The training data is:"
    print M.toarray()
    svd = TruncatedSVD(algorithm='randomized', n_components=6, n_iter=100)
    trained_data = svd.fit_transform(M, numpy.array(target))
    print "The transformed matrix is:"
    i = 0
    for rating_entry in trained_data[:10]:
        print "(user:"+uids[i]+", movie:"+mids[i]+")"
        print rating_entry
        print sum(rating_entry)
        i += 1
    #print trained_data
    return (trained_data, uids, mids)

def test(testing_data_file):
    (test_data, target, uids, mids) = load_data(testing_data_file)
    v = DictVectorizer()
    t = v.fit_transform((test_data))

    
(trained_data, uids, mids) = train('ml-100k/ua.base')
