import os
import collections
import math
import random
import numpy as np

import jieba
import tensorflow as tf

import typeahead_utils as util

#hanzi_unicode_regexp = re.compile(u"[^\u4E00-\u9FA5]+")

def read_files(root_dir):
    corpus = []
    i = 1
    for name in os.listdir(root_dir):
        filename = os.path.join(root_dir, name)
        if os.path.isfile(os.path.join(root_dir, name)):
                sys.stderr.write("process file %d %s\n"%(i, filename))

        f = open(filename)
        for line in f.readlines():
            seg_list = jieba.cut(line, hmm=False)
            corpus.extend(list(seg_list))
    return corpus


def build_dataset(corpus):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index  = 0

def generate_next_batch(batch_size, skip_window):
    global data, data_index
    batches = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    num_move = batch_size / skip_window
    for i in range(num_move):
        target_index = (data_index + skip_window) % len(data)
        for j in range(skip_window):
            labels[i * skip_window + j] = data[target_index]
            batches[i * skip_window + j] = data[(data_index + j) % len(data)] 
        data_index = (data_index + 1) % len(data)
    return batches, labels

def validate(valid_examples):
    global normalized_embeddings, reverse_dictionary
    global top_k
    example_embeddings = tf.nn.embedding_lookup(
            embeddings, valid_examples)
    sims = tf.matmul(example_embeddings, 
            noise_weights, transpose_b=True) + noise_biases
    similarity = sims.eval()
    for i, example_id in enumerate(valid_examples):
        sim = similarity[i]
        print sim
        ex_word = reverse_dictionary[example_id]
        nearest = (-sim).argsort()[:top_k]
        neighbors = map(lambda x : reverse_dictionary[x], nearest)
        print "%s nearest neighbor is %s" % (ex_word, neighbors) 


def predict(word):
    global dictionary
    word_id = dictionary[word]
    validate([word_id])

batch_size = 128
skip_window = 4
embedding_size = 128
vocabulary_size = 5000
num_sampled = 64
top_k = 8


filename = util.maybe_download('text8.zip', 31344016)

words = util.read_data(filename)
data, count, dictionary, reverse_dictionary = build_dataset(words)


x = tf.placeholder(tf.int32, shape=(batch_size))
y = tf.placeholder(tf.int32, shape=[batch_size])

embeddings = tf.Variable(tf.truncated_normal(
    shape=[vocabulary_size, embedding_size]))
embed = tf.nn.embedding_lookup(embeddings, x)

noise_weights = tf.Variable(tf.truncated_normal(
    shape = [vocabulary_size, embedding_size]))
noise_biases = tf.Variable(tf.zeros([vocabulary_size]))

pred_logits = tf.matmul(embed, noise_weights, transpose_b=True) \
    + noise_biases

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits, y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.2)
trainer = optimizer.minimize(loss)

norm = tf.sqrt(
        tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

test_set = ['happy', 'monday', 'university']
examples = map(lambda x: dictionary[x], test_set)

with sess.as_default():
    sess.run(init)

    for i in range(10000):
        batch_x, batch_y = generate_next_batch(batch_size, skip_window)
        _, cur_loss = sess.run([trainer, loss], feed_dict={x:batch_x, y:batch_y})
        if i % 1000 == 0:
            print "loss %f" % cur_loss
            validate(examples)
