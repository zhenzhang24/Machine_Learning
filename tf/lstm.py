import tensorflow as tf
import numpy as np
from random import shuffle


max_length = 20

def generate_training_data():
    train_input = ['{0:020b}'.format(i) for i in range(2**max_length)]
    shuffle(train_input)
    train_input = [map(int,i) for i in train_input]
    train_output = []
    ti  = []
    for i in train_input:
        temp_list = []
        count = 0
        for j in i:
            temp_list.append([j]) 
            if j == 1:
                count += 1
        ti.append(np.array(temp_list))
        train_output.append(count)
    train_input = ti
    return train_input, train_output


cur_index = 0
train_input, train_output = generate_training_data()

def generate_next_batch(batch_size):
    global cur_index, train_input, train_output
    data_size = len(train_input)
    end_index = (cur_index + batch_size) % data_size
    if end_index < cur_index:
        batch_x = train_input[cur_index:] + train_input[:end_index]
        batch_y = train_output[cur_index:] + train_output[:end_index]
    else:
        batch_x = train_input[cur_index:end_index]
        batch_y = train_output[cur_index:end_index]
    cur_index = end_index
    return batch_x, batch_y


batch_size = 100
num_units = 6
num_class = max_length + 1

x = tf.placeholder(tf.float32, [None, max_length, 1])
y = tf.placeholder(tf.int64, [None])



lstm = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
initial_state = lstm.zero_state(batch_size, tf.float32)

softmax_weight = tf.Variable(tf.truncated_normal(
    [num_units, num_class]))

softmax_biases = tf.Variable(tf.zeros([num_class]))

optimizer = tf.train.AdamOptimizer()

inputs = tf.unstack(x, num=max_length, axis=1)
outputs, final_state = tf.nn.rnn(lstm, inputs, dtype=tf.float32) #, initial_state=initial_state)
output = outputs[-1]

logits = tf.matmul(output, softmax_weight) + softmax_biases
loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
train = optimizer.minimize(loss)

mistakes = tf.not_equal(y, tf.argmax(logits, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

steps = 10000
with sess.as_default():
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y =  generate_next_batch(batch_size)
        _, cur_loss, mistake = sess.run(
                [train, loss, mistakes], feed_dict = {x:batch_x, y:batch_y})
        if i % 100 == 0:
            err = sess.run(error, 
                    feed_dict = {x:train_input, y:train_output})
            print "cur error is", err
