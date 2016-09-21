import tensorflow as tf
import numpy as np
from reader import Text
from tensorflow.python.ops import rnn_cell, rnn
from word2vec_api import W2V


def model(
    w2v: W2V,
    seq_length,
    batch_size,
    rnn_feature_size=100,
    num_hidden_layers=2,
    learning_rate=0.003
):
    x = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])
    y = tf.placeholder(tf.float32, [None, seq_length, w2v.width()])

    cell = rnn_cell.MultiRNNCell(
        [rnn_cell.BasicLSTMCell(rnn_feature_size)] * num_hidden_layers)

    W_ho = tf.Variable(tf.truncated_normal([rnn_feature_size, w2v.width()], stddev=0.1))

    b_ho = tf.Variable([[0.1] * w2v.width()])

    def nn(inputs, name, batch_size=batch_size, seq_length=seq_length):

        initial_state = cell.zero_state(batch_size, tf.float32)

        output, final_state = rnn.rnn(cell, inputs, initial_state=initial_state, scope=name)

        def logit(_x):
            return tf.matmul(_x, W_ho) + b_ho

        logits = [logit(tf.reshape(out, [seq_length, rnn_feature_size])) for out in tf.split(1, batch_size, output)]

        return initial_state, logits, final_state

    x_by_seq = [tf.reshape(vec, [batch_size, w2v.width()]) for vec in tf.split(1, seq_length, x)]
    initial_state, logits, final_state = nn(x_by_seq, name="rnn")

    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, logits))))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # needs to return input tensors (x, y, state) for feed_dict
    # as well as output tensors to train/ sample with. Train operations should probably be optional
    def train(sess, x_batch, y_batch):

        feed_dict = {x: x_batch, y: y_batch, initial_state: initial_state.eval()}
        _prediction, _cost, _final_state, _ = sess.run([logits, cost, final_state, optimizer], feed_dict=feed_dict)

        return _prediction, _cost, _final_state

    sample_x = tf.placeholder(tf.float32, [w2v.width()])
    sample_x_reshape = tf.reshape(sample_x, [1, w2v.width()])
    sample_init, sample_pred, sample_state = nn([sample_x_reshape], name='sample_rnn', batch_size=1, seq_length=1)

    def sample(sess, prime='The', words_to_sample=100):
        word_embeddings = [w2v.word_to_vector(prime)]
        _state = np.random.random(sample_init.get_shape())
        for i in range(words_to_sample):
            last_word = word_embeddings[i]
            _pred, _state = sess.run([sample_pred, sample_state],
                                     feed_dict={sample_x: last_word, sample_init: _state})
            word_embeddings.append(_pred[0][0])
        words = [w2v.vector_to_word(vec) for vec in word_embeddings]
        return words

    return train, sample


def train(word2vec: W2V, text: Text):
    seq_length = 100
    batch_size = 10

    train_batch, sample = model(word2vec, batch_size=batch_size, seq_length=seq_length)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(30000):
            x, y = text.batch(batch_size, seq_length)
            predication, cost, final_state = train_batch(sess, x, y)

            print(cost)
            if i % 10 == 0:

                predicted = [word2vec.vector_to_word(vec) for vec in predication[0]]
                y = [word2vec.vector_to_word(vec) for vec in y[0]]
                # words = [word2vec.vector_to_word(vec) for vec in predication[0]]
                print('actual0:\n' + str(predicted) + '\nactual1:\n' + str(y))
