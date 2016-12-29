from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def evaluate(output, obsv):
    pred = tf.argmax(output_logit, 1)

    correct = tf.equal(pred, obsv)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), reduction_indices=[0])
    return accuracy

mnist = input_data.read_data_sets("MNIST_data/")

input_dimension = 28 * 28
l1_units = 128 
l2_units = 32 
output_dimension = 10
learning_rate = 0.02
batch_size = 100
rounds = 10

x = tf.placeholder(tf.float32, shape=(None, input_dimension))
y_ = tf.placeholder(tf.int64, shape=(None))

# layer 1
weight_l1 = tf.Variable(tf.truncated_normal([input_dimension, l1_units]))
biases_l1 = tf.Variable(tf.zeros([l1_units]))

# layer 2
weight_l2 = tf.Variable(tf.truncated_normal([l1_units, l2_units]))
biases_l2 = tf.Variable(tf.zeros([l2_units]))

# output
weight_output = tf.Variable(tf.truncated_normal([l2_units, output_dimension]))
biases_output = tf.Variable(tf.zeros([output_dimension]))

# forward inference
l1_output = tf.nn.sigmoid(tf.matmul(x, weight_l1) + biases_l1)
l2_output = tf.nn.sigmoid((tf.matmul(l1_output, weight_l2) + biases_l2))
output_logit = tf.matmul(l2_output, weight_output) + biases_output

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    output_logit, y_)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

trainer = optimizer.minimize(loss)
evaluator = evaluate(output_logit, y_)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

num_examples = mnist.train.num_examples
steps = num_examples / batch_size
for r in range(rounds):
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, accuracy = sess.run([trainer, evaluator], 
                feed_dict={x:batch_x, y_:batch_y})

    accuracy = sess.run(evaluator, 
            feed_dict={x:mnist.test.images, y_:mnist.test.labels})
    print "evaluating accuracy", accuracy
