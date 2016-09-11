import itertools
import random
import tensorflow as tf
from argparse import ArgumentParser

# maybe saver
saver = None
parse = ArgumentParser()
parse.add_argument('-s', '--save')
args = parse.parse_args()

if args.save is not None:
    saver = tf.train.Saver()
    save_file = 'data/tfugh/' + args.save + '.ckpt'
    print('saving to ' + save_file)



# data
def times(num):
    def gen():
        i = -1
        while True:
            i += 1
            yield i * num
    return gen


fib_cache = {}


def fib():
    a, b = 0, 1
    yield a
    yield b
    while True:
        a, b = b, a + b
        yield b


def batch_generator(generators, training_seq_length, full_seq_size):
    print("creating batch generator")
    try:
        iter(generators)
        full_sequences = [list(itertools.islice(gen(), full_seq_size)) for gen in generators]
    except TypeError:
        full_sequences = [[itertools.islice(generators(), full_seq_size)]]

    max_elem = max([max(xs) for xs in full_sequences])
    full_sequences = [[x / max_elem for x in xs] for xs in full_sequences]

    def mk_batch(num_batches):
        def generate_sequence():
            elems = random.choice(full_sequences)
            base = random.randint(0, full_seq_size - training_seq_length - 1)
            return elems[base:base + training_seq_length], elems[base + training_seq_length:base + training_seq_length + 1]

        batches = [generate_sequence() for _ in range(num_batches)]
        return [x for x, _ in batches], [y for _, y in batches]
    print("made generator")

    def reconstitute(batch_result):
        return [[i * max_elem for i in seq] for seq in batch_result]

    return mk_batch, reconstitute






# Network Parameters
n_hidden_1 = 5
n_hidden_2 = 5
n_input = 3
n_output = 1

x = tf.placeholder("float", [None, n_input], name='x')
y = tf.placeholder("float", [None, n_output], name='y')




# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer




# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]), name='hout')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'out': tf.Variable(tf.random_normal([n_output]), name='bout')
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)






# Data Parameters
algo = [times(x) for x in range(1, 100)]
input_value_range = 2000

# Training Parameters
learning_rate = 0.0001
training_epochs = 100000
batch_per_epoch = 1000
batch_size = 1000
display_step = 5


# Define loss and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, pred))))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

mk_batch, reconstitute = batch_generator(algo, n_input, input_value_range)

# Launch the graph
init = tf.initialize_all_variables()


# Run
with tf.Session() as sess:
    if saver is not None:
        saver.restore(sess, save_file)
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        # Loop over all batches
        for batch in range(batch_per_epoch):
            batch_x, batch_y = mk_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            r, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / batch_per_epoch


        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", avg_cost)
            if epoch % (display_step * 5) == 0:
                if saver is not None:
                    saver.save(sess, save_file)
                n_y = tf.identity(y)
                test_x, test_y = mk_batch(1)
                print("for input", reconstitute(test_x))
                print("and ouput", reconstitute(test_y))
                print("got", reconstitute(sess.run(pred, {x: test_x, y: test_y})))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x, test_y = mk_batch(batch_size)
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))


