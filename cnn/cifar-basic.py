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

    # input shape [batch, in_height, in_width, in_channels] and 
    # a filter shape [filter_height, filter_width, 
    # in_channels, out_channels],
    def conv(self, input_fm):
        return tf.nn.relu(tf.nn.conv2d(input_fm, self.W, 
                                    strides=[1, stride, stride, 1], 
                                    padding='SAME')
                        + self.biases)
        

class MaxPoolLayer:
    def __init__(self, ksize, stride):
        self.ksize = ksize
        self.strides = stride

    def pool(self, input_fm):
        pooling_feature_map = tf.nn.max_pool(input_fm, 
                ksize=[1, self.ksize, self.ksize, 1],
                strides=[1, self.strides, self.strides, 1], padding='SAME')

        return pooling_feature_map

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
batch_size = 100
rounds = 200
output_dimension = 10

train_input, train_output = input_data.read_data_sets(
    "/Users/zhenzhang/Documents/ML/data/cifar-10/data_batch_1")

x = tf.placeholder(tf.float32, 
        shape=(None, image_height, image_width, initial_num_features))
y_ = tf.placeholder(tf.int64, shape=(None))

filter_shape = (3, 3)
in_channels = 3
out_channels = 32
stride = 1
conv_1 = ConvLayer(in_channels, out_channels, filter_shape, stride)
conv_1_output = conv_1.conv(x)
pool_1 = MaxPoolLayer(3, 2)
pool_1_output = pool_1.pool(conv_1_output)


filter_shape = (3, 3)
in_channels = out_channels
out_channels = 64
stride = 1
conv_2 = ConvLayer(in_channels, out_channels, filter_shape, stride)
conv_2_output = conv_2.conv(pool_1_output)
pool_2 = MaxPoolLayer(3, 2)
pool_2_output = pool_2.pool(conv_2_output)

filter_shape = (3, 3)
in_channels = out_channels
out_channels = 128
stride = 1
conv_3 = ConvLayer(in_channels, out_channels, filter_shape, stride)
conv_3_output = conv_3.conv(pool_2_output)
pool_3 = MaxPoolLayer(3, 1)
pool_3_output = pool_3.pool(conv_3_output)

fc_1_num_features = 256
shape = pool_3_output.get_shape()
fc_1_input_shape = int(shape[1] * shape[2] * shape[3])
fc_1 = FCLayer(fc_1_input_shape, fc_1_num_features)
fc_1_output = fc_1.matmul(pool_3_output)

fc_2_num_features = 256
fc_2 = FCLayer(fc_1_num_features, fc_2_num_features)
fc_2_output = fc_2.matmul(fc_1_output)


fc_output_num_features = output_dimension
fc_output = FCLayer(fc_2_num_features, fc_output_num_features, False)
fc_output_output = fc_output.matmul(fc_2_output)

keep_prob = tf.placeholder(tf.float32)
pred_logits = fc_output_output

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits, y_) 
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
trainer = optimizer.minimize(loss)
evaluator = evaluate(pred_logits, y_)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_examples = 10000
steps = num_examples / batch_size
for r in range(rounds):
    for i in range(steps):
        batch_x, batch_y = generate_next_batch(batch_size)
        _, x_in, output, accuracy = sess.run(
                [trainer, x, pred_logits, evaluator],
                feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
        if i % 100 == 0:
            x_in, output, c_1, p_1, c_2, p_2,  c_3, p_3 = sess.run(
                    [x, pred_logits, conv_1_output, pool_1_output, 
                        conv_2_output, pool_2_output,
                        conv_3_output, pool_3_output,
                        ],
                    feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
            print "input shape", x_in.shape
            print "output shape", output.shape
            print c_1.shape
            print p_1.shape
            print c_2.shape
            print p_2.shape
            print c_3.shape
            print p_3.shape

            accuracy = sess.run(evaluator, 
                feed_dict={x:train_input, y_:train_output, keep_prob:0.5})
            print "evaluating accuracy %d steps: %f" % (r*steps + i, accuracy)
