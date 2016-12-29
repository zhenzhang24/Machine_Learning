import math
import tensorflow as tf
import numpy as np

NUM_CLASSES = 1

def loss(pred, real):
    cross_entrophy = -1 * (real * tf.log(pred) + (1-real) * tf.log(1-pred))
    loss = tf.reduce_mean(cross_entrophy, name='xentrophy_mean')
    return loss

def train(loss_func, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss_func, global_step=global_step)
    return train_op

def evaluate(pred, real):
    pred_mapped = tf.round(pred)
    correct = tf.equal(pred_mapped, real)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), 
            reduction_indices=[0])
    return (pred_mapped, accuracy)

def fill_feed_dict(X_pl, Y_pl):
    x = np.array([[0,0], [0,1], [1,0,],[1,1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return {X_pl:x, Y_pl:y}

input_dimension = 2
l1_units = 2
l2_units = 1
X = tf.placeholder(tf.float32, shape=(None, input_dimension))
Y = tf.placeholder(tf.float32, shape=(None, l2_units))

(n_samples, n_features) = X.get_shape()
weights_l1 = tf.get_variable('weights_l1',
        shape=[n_features.value, l1_units], 
        initializer=tf.random_uniform_initializer())
biases_l1 = tf.get_variable('biases_l1', 
        initializer=tf.zeros_initializer([l1_units]))

hidden_l1 = tf.sigmoid(tf.matmul(X, weights_l1) + biases_l1)

weights_l2 = tf.get_variable('weights_l2',
        shape=[l1_units, l2_units],
        initializer=tf.random_uniform_initializer())
biases_l2 = tf.get_variable('biases_l2',
        initializer=tf.zeros_initializer([l2_units]))

pred = tf.sigmoid(tf.matmul(hidden_l1, weights_l2) + biases_l2)


loss = loss(pred, Y)
train_op = train(loss, 0.02)
eval_op = evaluate(pred, Y)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
feed_dict = fill_feed_dict(X, Y)
for i in range(100000):
    sess.run(train_op, feed_dict=feed_dict)
    if i % 10000 == 0:
        print(sess.run(eval_op, feed_dict=feed_dict))
        print(sess.run(loss, feed_dict=feed_dict))
        print(sess.run(weights_l1))
        print(sess.run(weights_l2))
