from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import input_data

class ConvLayer:
    def __init__(self, num_input_fm, num_filter, filter_shape, stride):
        self.num_input_feature_map = num_input_fm
        self.num_filter = num_filter
        self.filter_shape = filter_shape

        w_shape = (filter_shape[0], filter_shape[1], num_input_fm, num_filter)
        self.W = tf.Variable(tf.truncated_normal(w_shape,
            stddev = 0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[num_filter]))
        self.stride = stride

    # input shape [batch, in_height, in_width, in_channels] and 
    # a filter shape [filter_height, filter_width, 
    # in_channels, out_channels],
    def conv(self, input_fm):
        return tf.nn.relu(tf.nn.conv2d(input_fm, self.W, 
                                    strides=[1, self.stride, self.stride, 1], 
                                    padding='SAME')
                        + self.biases)

    def lrn(self, relu_res):
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn_res = tf.nn.local_response_normalization(relu_res, 
                depth_radius=radius, 
                alpha=alpha, 
                beta=beta, 
                bias=bias)
        return lrn_res
        
    def compute(self, input_fm):
        conv_res = self.conv(input_fm)
        return self.lrn(conv_res)

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
        res = ConvLayer(in_channels, out_channels, (ksize, ksize), strides)
        return res
    if layer['type'] == 'pool':
        (ksize, strides) = layer['config']
        return MaxPoolLayer(ksize, strides)

cur_index = 0
def generate_next_batch(batch_size):
    global cur_index, train_input, train_output
    data_size = len(train_input)
    end_index = (cur_index + batch_size) % data_size
    if end_index < cur_index:
        batch_x = np.concatenate(
                (train_input[cur_index:], train_input[:end_index]))
        batch_y = np.concatenate(
                (train_output[cur_index:], train_output[:end_index]))
    else:
        batch_x = train_input[cur_index:end_index]
        batch_y = train_output[cur_index:end_index]
        cur_index = end_index
    return batch_x, batch_y

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
rounds = 200
output_dimension = 10

train_input, train_output = input_data.read_data_sets(
    "/Users/zhenzhang/Documents/ML/data/cifar-10/data_batch_1")

x = tf.placeholder(tf.float32, 
        shape=(None, image_height, image_width, initial_num_features))
y_ = tf.placeholder(tf.int64, shape=(None))

configs = []
configs.append({'type':'conv', 'config':(3, 32, 3, 1)})
#configs.append({'type':'conv', 'config':(32, 32, 3, 1)})
configs.append({'type':'pool', 'config':(3, 2)})

configs.append({'type':'conv', 'config':(32, 64, 3, 1)})
#configs.append({'type':'conv', 'config':(64, 64, 3, 1)})
configs.append({'type':'pool', 'config':(3, 2)})

configs.append({'type':'conv', 'config':(64, 128, 3, 1)})
configs.append({'type':'pool', 'config':(3, 1)})

network = CNNNet(configs)
(cnn_output, cnn_structure) = network.forward(x)

fc_1_num_features = 256
shape = cnn_output.get_shape()
fc_1_input_shape = int(np.prod(shape[1:]))
fc_1 = FCLayer(fc_1_input_shape, fc_1_num_features).matmul(cnn_output)

fc_2_num_features = 256
fc_2 = FCLayer(fc_1_num_features, fc_2_num_features).matmul(fc_1)


fc_output_num_features = output_dimension
fc_output = FCLayer(fc_2_num_features, fc_output_num_features, False).matmul(fc_2)

keep_prob = tf.placeholder(tf.float32)
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
for r in range(rounds):
    for i in range(steps):
        batch_x, batch_y = generate_next_batch(batch_size)
        _, x_in, output, accuracy = sess.run(
                [trainer, x, pred_logits, evaluator],
                feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
        if i % 100 == 0:
            x_in, output, structure= sess.run(
                    [x, pred_logits, cnn_structure],
                    feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
            print "input shape", x_in.shape
            print "output shape", output.shape
            for l in structure:
                print l

            accuracy = sess.run(evaluator, 
                feed_dict={x:train_input, y_:train_output, keep_prob:0.5})
            print "evaluating accuracy %d steps: %f" % (r*steps + i, accuracy)
