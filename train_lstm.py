import tensorflow as tf
from reader import Text
from tensorflow.python.ops import rnn_cell, rnn
from word2vec_api import W2V


def model(
    w2v: W2V,
    seq_length,
    batch_size,
    rnn_feature_size=300,
    num_hidden_layers=2,
    learning_rate=0.01
):

    x = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])
    y = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])

    cell = rnn_cell.MultiRNNCell(
        [rnn_cell.BasicLSTMCell(rnn_feature_size)] * num_hidden_layers)
    # cell = rnn_cell.BasicLSTMCell(rnn_feature_size)

    initial_state = cell.zero_state(batch_size, tf.float32)

    prediction, final_state = rnn.dynamic_rnn(cell, x, dtype=tf.float32)

    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # needs to return input tensors (x, y, state) for feed_dict
    # as well as output tensors to train/ sample with. Train operations should probably be optional
    def train(sess, x_batch, y_batch):
        feed_dict = {
            x: x_batch, y: y_batch, initial_state: initial_state.eval()
        }
        _prediction, _cost, _final_state, _ = sess.run([prediction, cost, final_state, optimizer], feed_dict=feed_dict)

        return _prediction, _cost, _final_state

    def sample(sess):
        raise NotImplemented("sample")

    return train, sample


def train(word2vec: W2V, text: Text):
    seq_length = 50
    batch_size = 51

    train_batch, sample = model(word2vec, seq_length=seq_length, batch_size=batch_size)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(100):
            x, y = text.batch(batch_size, seq_length)
            predication, cost, final_state = train_batch(sess, x, y)
            print(cost)

