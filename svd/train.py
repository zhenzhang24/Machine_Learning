import sys
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
import pyfm
from pyfm import pylibfm
import numpy as np

g_occupation_list = []
g_user_attrs_list = []
g_num_users = 943
g_num_movies = 1682
g_users = []
g_movies = []

#User_Attrs
ATTR_UID = 0
ATTR_AGE = 1
ATTR_GENDER = 2
ATTR_OCCUPATION = 3
ATTR_ZIPCODE = 4

#Move_Attrs
ATTR_MOVIE_ID = 0
ATTR_TITLE = 1
ATTR_RELEASE_DATE = 2
ATTR_VIDEO_RELEASE_DATE = 3
ATTR_URL = 4
ATTR_GENRE = 5


class User:

    def __init__(self, attrs):
        self.uid = attrs[ATTR_UID]
        self.age = int(attrs[ATTR_AGE])
        self.gender = attrs[ATTR_GENDER]
        self.occupation = attrs[ATTR_OCCUPATION]
        self.zipcode = attrs[ATTR_ZIPCODE]
        self.age_group = min(self.age / 10, 6)

class Movie:

    def __init__(self, attrs):
        self.id = int(attrs[ATTR_MOVIE_ID])
        self.title = attrs[ATTR_TITLE]
        self.release_date = attrs[ATTR_RELEASE_DATE]
        self.video_release_date = attrs[ATTR_VIDEO_RELEASE_DATE]
        self.url = attrs[ATTR_URL]
        self.genre = attrs[ATTR_GENRE:]
        # to do: add recency to movie attrs


def build_all_users(u_file_name):
    f = open(u_file_name)
    users = {}
    for line in f.readlines():
        attrs = line.split('|')
        user = User(attrs)
        users[user.uid] = user
    f.close()
    return users

def build_all_movies(m_file_name):
    f = open(m_file_name)
    movies = []
    for line in f.readlines():
        fields = line.split('|')
        movie = Movie(fields)
        movies.append(movie)
    f.close()
    return movies

def build_user_attr_list(occupation_file):
    f = occupation_file.open()
    g_occupation_list = f.readlines()
    f.close()

    g_user_attrs_list = {'gender_M':0, 'gender_F':1}
    ind = 2
    for i in range(6):
        g_user_attrs_list['age_group_'+str(i)] = ind
        ind += 1
    for occu in g_occupation_list:
        g_user_attrs_list['occupation_'+occu] = ind
        ind += 1

class MySVD:
    def __init__(self, training_data_file, num_components=100):
        self.M = self.build_training_data(training_data_file)
        self.svd = TruncatedSVD(n_components=num_components)


    def train(self):
        self.U = self.svd.fit_transform(self.M)
        self.V = self.svd.components_
        self.M_predicted = np.dot(self.U, self.V)

    def build_training_data(self, training_data_file):
        f = open(training_data_file)
        training_data = []
        row_indices = []
        column_indices = []
        for line in f.readlines():
            entry = line.split('\t')
            uid = int(entry[0])
            mid = int(entry[1])
            rating = int(entry[2])
            timestamp = int(entry[3])
            
            user = users[uid-1]
            movie = movies[mid-1]
            row_indices.append(uid-1)
            column_indices.append(mid-1)
            training_data.append(rating)
        X = csc_matrix((training_data, 
            (np.array(row_indices), np.array(column_indices))),
            shape=(g_num_users, g_num_movies))
        f.close()
        return X


    def predict(self, uid, vid):
        return self.M_predicted[uid-1, vid-1]

    def test(self, test_data):
        pass

class MyFM:
    def __init__(self, n_factor=16, n_iter=10, use_attrs=True):
        self.fm = pylibfm.FM(num_factors=n_factor, 
                num_iter=n_iter,
                task="regression",
                initial_learning_rate=0.001,
                learning_rate_schedule="optimal")
        self.use_attrs = use_attrs
        self.v = DictVectorizer()

    def load_data(self, file_name):
        f = open(file_name)
        data = []
        target = []
        for line in f.readlines():
            entry = line.split('\t')
            features = {}
            features['uid'] = uid = entry[0]
            features['mid'] = entry[1]
            if self.use_attrs and uid in g_users:
                user = g_users[uid]
                features['gender'] = user.gender
                features['occupation'] = user.occupation
            data.append(features)
            target.append(float((entry[2])))

        return (data, target)

    def train(self, training_data_file):
        f = open(training_data_file)
        (self.data, self.target) = self.load_data(training_data_file)

        self.X = self.v.fit_transform(self.data)
        self.Y = np.array(self.target)
        print self.X.shape
        return self.fm.fit(self.X, self.Y)

    def test(self, test_file):
        (test_data, test_target) = self.load_data(test_file)
        test_result = self.predict(test_data)
        print test_result
        return (test_result, test_target)

    def predict(self, test_data):
        sample = self.v.transform(test_data)
        return self.fm.predict(sample)

def calc_reverse_order(pred_val, real_val):
    orders = []
    for i in range(len(real_val)):
        orders.append((pred_val[i], real_val[i]))
    return inverse_ratio(orders)

def inverse_ratio(order):
    # in y_, -rating ascendant order.
    order = sorted(order, key=lambda x:(x[0], -x[1]))

    prev_count = {} 
    inverse_count = 0 
    total_count = 0 
    for i, (_, r) in enumerate(order): 
        for key, count in prev_count.iteritems():
            total_count += count
            if key > r: inverse_count += count
                                                                                    # update previous count 
        prev_count[r] = prev_count.get(r, 0) + 1
    inv_ratio = inverse_count * 100.  / total_count
    return inv_ratio, inverse_count, total_count

g_users = build_all_users('ml-100k/u.user')
print "totla number of users:", len(g_users)
g_movies = build_all_movies('ml-100k/u.item')
print "total number of movies:", len(g_movies)
g_num_users = len(g_users)
g_num_movies = len(g_movies)

#svd = MySVD('ml-100k/ua.base')
#svd.train()
#svd.predict(1,1)

training_data_file = sys.argv[1]
test_data_file = sys.argv[2]
iters = int(sys.argv[3])
use_attrs = True if sys.argv[4] == 'Y' else False
fm = MyFM(n_iter=iters, use_attrs=use_attrs)
fm.train(training_data_file)
print "Config: "
print "training_data: ", training_data_file
print "testing data:", test_data_file
print "n_iters:", iters
print "use_attrs:", use_attrs 
(predicted, test_y) = fm.test(test_data_file)
reverse_order = calc_reverse_order(predicted, test_y)
print "#reverse pairs:", reverse_order

# use data set ua.base, ua.test
# iter=1, use_attr=False, ratio = 36.798"
# iter=1, use_attr=True, ration = 36.539"
# iter=10, use_attr=False, ratio = 36.98"
# iter=10, use_attr=True, ration = 36.774"
# iter=100, use_attr=False, ratio = 37.24"
# iter=100, use_attr=True, ratio = 19.57"

# use data set u1.base, u1.test)

