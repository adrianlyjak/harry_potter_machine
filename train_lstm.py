import tensorflow as tf
import numpy as np
import random
from word2vec_api import W2V
from tensorflow.python.ops import rnn_cell, rnn
from reader import Text
import os

def run(word2vec: W2V, text: Text):

    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 10
    display_step = 10

    # Network Parameters
    n_vocab_chars = word2vec.width()
    n_seq_length = 20  # characters to sample
    n_hidden = 30  # hidden layer num of features
    n_classes = word2vec.width()

    # tf Graph input
    x = tf.placeholder("float", [None, n_seq_length, n_vocab_chars])
    # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
    istate = tf.placeholder("float", [None, 2 * n_hidden])
    y = tf.placeholder("float", [None, n_seq_length, n_classes])

    # Define weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_vocab_chars, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def prediction_graph(_X, _istate, _weights, _biases):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, n_vocab_chars])  # (n_steps*batch_size, n_input)
        # Linear activation
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, n_seq_length, _X)  # n_steps * (batch_size, n_hidden)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

        # Linear activation
        # Get inner loop last output
        return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    def flatten_sequences(_y):
        # input shape: (batch_size, n_steps, n_input)
        _y = tf.transpose(_y, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _y = tf.reshape(_y, [-1, n_vocab_chars])  # (n_steps*batch_size, n_input)

        return _y

    print("get predictor")
    pred = prediction_graph(x, istate, weights, biases)
    flat_y = flatten_sequences(y)

    print("get cost/optimizer")
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, flat_y))  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    print("get correct_pred/accuracy")
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(flat_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("get init")
    # Initializing the variables
    init = tf.initialize_all_variables()

    print("launch session")
    # Launch the graph
    with tf.Session() as sess:
        print("session launched; run init ...")
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = txt.batch(batch_size, n_seq_length)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           istate: np.zeros((batch_size, 2 * n_hidden))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                    istate: np.zeros((batch_size, 2 * n_hidden))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                                 istate: np.zeros((batch_size, 2 * n_hidden))})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")


