from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import random
import input_data

class ConvLayer:
    def __init__(self, num_input_fm, num_filter, filter_shape, stride, dropout):
        self.num_input_feature_map = num_input_fm
        self.num_filter = num_filter
        self.filter_shape = filter_shape

        w_shape = (filter_shape[0], filter_shape[1], num_input_fm, num_filter)
        self.W = tf.Variable(tf.truncated_normal(w_shape,
            stddev = 0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[num_filter]))
        self.stride = stride
        self.dropout = dropout

    # input shape [batch, in_height, in_width, in_channels] and 
    # a filter shape [filter_height, filter_width, 
    # in_channels, out_channels],
    def conv(self, input_fm):
        return tf.nn.relu(tf.nn.conv2d(input_fm, self.W, 
                                    strides=[1, self.stride, self.stride, 1], 
                                    padding='VALID')
                        + self.biases)

       
    def compute(self, input_fm):
        conv_res = self.conv(input_fm)
        conv_res = tf.nn.dropout(conv_res, self.dropout)
        return conv_res

class MaxPoolLayer:
    def __init__(self, ksize, stride):
        self.ksize = ksize
        self.strides = stride

    def pool(self, input_fm):
        pooling_feature_map = tf.nn.max_pool(input_fm, 
                ksize=[1, self.ksize, self.ksize, 1],
                strides=[1, self.strides, self.strides, 1], padding='SAME')

        return pooling_feature_map

    def compute(self, input_fm):
        return self.pool(input_fm)

class LRN: 
    def __init__(self):
        pass

    def compute(self, relu_res):
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn_res = tf.nn.local_response_normalization(relu_res, 
                depth_radius=radius, 
                alpha=alpha, 
                beta=beta, 
                bias=bias)
        return lrn_res
 
class FCLayer:
    def __init__(self, input_size, out_features, relu=True):
        self.input_size = input_size
        w_shape = (self.input_size, out_features)
        self.W = tf.Variable(
                tf.truncated_normal(w_shape, stddev = 0.1))
        self.biases = tf.Variable(tf.zeros([out_features])) 
        self.relu = relu

    def matmul(self, input_fm):
        flat = tf.reshape(input_fm, [-1, self.input_size])
        if self.relu:
            return  tf.nn.relu(tf.matmul(flat, self.W) + self.biases)
        else:
            return  tf.matmul(flat, self.W) + self.biases

class CNNNet:
    def __init__(self, config):
        self.layers = []
        self.layered_output = []
        self.dimensions = []
        for layer_conf in config:
            layer = initiate_layer(layer_conf)
            self.layers.append(layer)

    def forward(self, input_data):
        #self.layered_output.append(input_data)
        self.dimensions = []
        self.dimensions.append(tf.shape(input_data))
        cur_output = input_data
        for layer in self.layers:
            cur_output = layer.compute(cur_output)
            self.dimensions.append(tf.shape(cur_output))
            #self.layered_output.append(cur_output)
        return (cur_output, self.dimensions)

    def get_structure(self):
        return self.dimensions

def initiate_layer(layer):
    if layer['type'] == 'conv':
        (in_channels, out_channels, ksize, strides)= layer['config']
        if layer.has_key('dropout'):
            dropout = layer['dropout']
        else:
            dropout = 1
        res = ConvLayer(in_channels, out_channels, 
                (ksize, ksize), strides, dropout)
        return res
    if layer['type'] == 'pool':
        (ksize, strides) = layer['config']
        return MaxPoolLayer(ksize, strides)

    if layer['type'] == 'lrn':
        return LRN()

cur_index = 0
ids = []
def generate_next_batch2(batch_size):
    global cur_index, train_input, train_output
    ids = random.sample(range(len(train_input)), batch_size)
    batch_x = []
    batch_y = []
    for id in ids:
        batch_x.append(train_input[id])
        batch_y.append(train_output[id])
    return np.array(batch_x), np.array(batch_y)

def generate_next_batch(batch_size):
    global cur_index, train_input, train_output, ids
    data_size = len(train_input)
    if len(ids) == 0:
        ids = range(data_size)
    end_index = (cur_index + batch_size) % data_size
    if end_index < cur_index:
        batch_ids = ids[cur_index:] + ids[:end_index]
   #     random.shuffle(ids)
    else:
        batch_ids = ids[cur_index:end_index]
    cur_index = end_index
    batch_x = []
    batch_y = []
    for id in batch_ids:
        batch_x.append(train_input[id])
        batch_y.append(train_output[id])
    return np.array(batch_x), np.array(batch_y)

def evaluate(output_logit, obsv):
    pred = tf.argmax(output_logit, 1)
    correct = tf.equal(pred, obsv)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), reduction_indices=[0])
    return accuracy


image_width = 32
image_height = 32
initial_num_features = 3
learning_rate = 0.001
batch_size = 50
rounds = 20000
output_dimension = 10

#train_input, train_output = input_data.read_data_sets(
#    "/Users/zhenzhang/Documents/ML/data/cifar-10/data_batch_1")

test_input, test_output = input_data.read_data_sets(
    "/Users/zhenzhang/Documents/ML/data/cifar-10/test_batch")
x = tf.placeholder(tf.float32, 
        shape=(None, image_height, image_width, initial_num_features))
y_ = tf.placeholder(tf.int64, shape=(None))
dropout = tf.placeholder(tf.float32)

configs = []
configs.append({'type':'conv', 'config':(3, 32, 2, 1), 'dropout':dropout})
configs.append({'type':'pool', 'config':(3, 2)})
configs.append({'type':'lrn'})

configs.append({'type':'conv', 'config':(32, 64, 2, 1), 'dropout':dropout})
configs.append({'type':'pool', 'config':(3, 2)})
configs.append({'type':'lrn'})

configs.append({'type':'conv', 'config':(64, 64, 2, 1), 'dropout':dropout})
configs.append({'type':'pool', 'config':(3, 1)})
configs.append({'type':'lrn'})

configs.append({'type':'conv', 'config':(64, 64, 2, 1), 'dropout':dropout})
configs.append({'type':'pool', 'config':(3, 1)})
configs.append({'type':'lrn'})

configs.append({'type':'conv', 'config':(64, 128, 2, 1)})
configs.append({'type':'pool', 'config':(3, 1)})
configs.append({'type':'lrn'})

network = CNNNet(configs)
(cnn_output, cnn_structure) = network.forward(x)

fc_1_num_features = 512
shape = cnn_output.get_shape()
fc_1_input_shape = int(np.prod(shape[1:]))
fc_1 = FCLayer(fc_1_input_shape, fc_1_num_features).matmul(cnn_output)

fc_2_num_features = 1024
fc_2 = FCLayer(fc_1_num_features, fc_2_num_features).matmul(fc_1)

fc_output_num_features = output_dimension
fc_output = FCLayer(fc_2_num_features, fc_output_num_features, False).matmul(fc_2)

pred_logits = fc_output

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits, y_) 
optimizer = tf.train.AdamOptimizer(learning_rate)
trainer = optimizer.minimize(loss)
evaluator = evaluate(pred_logits, y_)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_examples = 10000
steps = num_examples / batch_size
print configs
fno = 0
for i in range(rounds):
        if i % 1000 == 0:
            fno = (fno) % 5 + 1
            filename = "/Users/zhenzhang/Documents/ML/data/cifar-10/data_batch_"+str(fno)
            print filename
            train_input, train_output = input_data.read_data_sets(
                    filename)
        batch_x, batch_y = generate_next_batch(batch_size)
        _, x_in, output, accuracy = sess.run(
                [trainer, x, pred_logits, evaluator],
                feed_dict={x:batch_x, y_:batch_y, dropout:0.8})
        if i == 0:
            x_in, output, structure= sess.run(
                    [x, pred_logits, cnn_structure],
                    feed_dict={x:batch_x, y_:batch_y, dropout:0.8})
            print "input shape", x_in.shape
            print "output shape", output.shape
            for l in structure:
                print l

        if i % 100 == 0:
            accuracy = sess.run(evaluator, 
                    feed_dict={x:test_input[:400], y_:test_output[:400], dropout:1.0})
            print "evaluating accuracy %d steps: %f" % (i, accuracy)

