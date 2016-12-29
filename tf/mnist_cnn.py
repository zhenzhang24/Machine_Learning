from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class CNNLayer:
    def __init__(self, num_input_fm, num_filter, filter_shape):
        self.num_input_feature_map = num_input_fm
        self.num_filter = num_filter
        self.filter_shape = filter_shape

        w_shape = (filter_shape[0], filter_shape[1], num_input_fm, num_filter)
        self.W = tf.Variable(tf.truncated_normal(w_shape,
            stddev = 0.1))
        self.biases = tf.Variable(tf.zeros(num_filter))

    # input shape [batch, in_height, in_width, in_channels] and 
    # a filter shape [filter_height, filter_width, 
    # in_channels, out_channels],
    def conv_and_pool(self, input_fm):
        self.cnn_feature_map = \
            tf.nn.relu(tf.nn.conv2d(input_fm, self.W, 
                                    strides=[1, 1, 1, 1], padding='SAME')
                        + self.biases)
        self.pooling_feature_map = tf.nn.max_pool(self.cnn_feature_map, 
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
        return self.pooling_feature_map

def evaluate(output_logit, obsv):
    pred = tf.argmax(output_logit, 1)
    correct = tf.equal(pred, obsv)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), reduction_indices=[0])
    return accuracy


image_width = 28
image_height = 28
learning_rate = 0.0002
batch_size = 100
rounds = 5
output_dimension = 10

l1_num_filters = 32
l2_num_filters = 64
fc1_units = 1024
fc1_units = output_dimension
fc2_units = output_dimension

mnist = input_data.read_data_sets("MNIST_data/")

images = tf.placeholder(tf.float32, shape=(None, image_width*image_height))
y_ = tf.placeholder(tf.int64, shape=(None))
x = tf.reshape(images, [-1, image_width, image_height, 1])

filter_shape = (5, 5)
layer1 = CNNLayer(1, l1_num_filters, filter_shape)
layer1_output = layer1.conv_and_pool(x)

layer2 = CNNLayer(l1_num_filters, l2_num_filters, filter_shape)
layer2_output = layer2.conv_and_pool(layer1_output)

l2_feature_map_size = 28/2/2*28/2/2
fc1_shape = [l2_feature_map_size * l2_num_filters, fc1_units]
fc1_weights = tf.Variable(tf.truncated_normal(fc1_shape))
fc1_biases = tf.Variable(tf.zeros([fc1_units])) 
layer2_flat = tf.reshape(layer2_output, 
        [-1, l2_feature_map_size * l2_num_filters])
fc1_output = tf.nn.relu(tf.matmul(layer2_flat, fc1_weights) + fc1_biases)

fc2_shape = [fc1_units, fc2_units]
fc2_weights = tf.Variable(tf.truncated_normal(fc2_shape))
fc2_biases = tf.Variable(tf.zeros([fc2_units]))
fc2_output = tf.nn.relu(tf.matmul(fc1_output, fc2_weights) + fc2_biases)

pred_logits = fc1_output

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logits, y_) 
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
trainer = optimizer.minimize(loss)
evaluator = evaluate(pred_logits, y_)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_examples = mnist.train.num_examples
steps = num_examples / batch_size
for r in range(rounds):
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, image_in, x_in, output, accuracy = sess.run(
                [trainer, images, x, pred_logits, evaluator],
            feed_dict={images:batch_x, y_:batch_y})
    accuracy = sess.run(evaluator,
            feed_dict={images:mnist.test.images, y_:mnist.test.labels})
    print "evaluating accuracy", accuracy
