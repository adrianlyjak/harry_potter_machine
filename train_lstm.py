import tensorflow as tf
import numpy as np
from reader import Text
from tensorflow.python.ops import rnn_cell, rnn
from word2vec_api import W2V


def model(
    w2v: W2V,
    seq_length,
    batch_size,
    rnn_feature_size=300,
    num_hidden_layers=2,
    learning_rate=0.003
):
    x = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])
    y = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])

    cell = rnn_cell.MultiRNNCell(
        [rnn_cell.BasicLSTMCell(rnn_feature_size)] * num_hidden_layers)

    initial_state = cell.zero_state(batch_size, tf.float32)

    prediction, final_state = rnn.dynamic_rnn(cell, x, initial_state=initial_state)

    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # needs to return input tensors (x, y, state) for feed_dict
    # as well as output tensors to train/ sample with. Train operations should probably be optional
    def train(sess, x_batch, y_batch):
        # rearrange input batches from [batch_size x seq_length x n_features]
        #                           to [seq_length x batch_size x n_features]
        # x_batch = np.transpose(x_batch, [1,0,2])
        # y_batch = np.transpose(y_batch, [1,0,2])

        feed_dict = {
            x: x_batch, y: y_batch, initial_state: initial_state.eval()
        }
        _prediction, _cost, _final_state, _ = sess.run([prediction, cost, final_state, optimizer], feed_dict=feed_dict)

        return _prediction, _cost, _final_state

    sample_x = tf.placeholder(tf.float32, [1, w2v.width()], "sample_x")
    sample_initial_state = tf.placeholder(tf.float32, [1, initial_state.get_shape().dims[1]], name="sample_state")
    sample_pred, sample_state = rnn.rnn(cell, [sample_x], initial_state=sample_initial_state, scope="sample_rnn")

    def sample(sess, prime='The', words_to_sample=100):
        word_embeddings = [w2v.word_to_vector(prime)]
        _state = np.zeros(sample_initial_state.get_shape())
        for i in range(words_to_sample):
            last_word = word_embeddings[i:i + 1]
            _pred, _state = sess.run([sample_pred, sample_state],
                                     feed_dict={sample_x: last_word, sample_initial_state: _state})
            word_embeddings.append(_pred[0][0])
        words = [w2v.vector_to_word(vec) for vec in word_embeddings]
        return words

    return train, sample


def train(word2vec: W2V, text: Text):
    seq_length = 50
    batch_size = 51

    train_batch, sample = model(word2vec, batch_size=batch_size, seq_length=seq_length)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(100):
            x, y = text.batch(batch_size, seq_length)
            predication, cost, final_state = train_batch(sess, x, y)
            print(cost)
            if i % 10 == 0:
                print(' '.join(sample(sess, words_to_sample=50)))
