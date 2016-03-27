"""
This program analyze the integer sequence using Bi-diractional Recurrent Neural
Network (BRNN) with Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
based on the python library Keras.

"Keras is a minimalist, highly modular neural networks library, written in
 Python and capable of running on top of either TensorFlow or Theano."

                                                ---- Keras (http://keras.io/)


It is based on this Keras example - imdb_bidirectional_lstm.py:
https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

Author: Chang Liu (fluency03)
Data: 2016-03-26
"""


from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Graph
from keras.utils.np_utils import accuracy
from keras.utils.visualize_util import plot
import numpy as np
import sys

# random number generator with a fixed value for reproducibility
np.random.seed(1337)



class SequenceAnalyzer(object):
    """
    A integer sequence analyzer. RNN Graph model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len,
                 return_sequence=True):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.return_sequence = return_sequence
        self.model = Graph()

    def build_lstm(self, dropout=0.2):
        """
        Bidirectional LSTM with specified dropout rate, a model built with
        softmax activation, cross entropy loss and rmsprop optimizer
        """
        self.model.add_input(input_shape=(self.sentence_length, self.input_len),
                             name='input', dtype='float')
        self.model.add_node(LSTM(self.hidden_len),
                            name='forward', input='input')
        self.model.add_node(LSTM(self.hidden_len, go_backwards=True),
                            name='backward', input='input')
        self.model.add_node(Dropout(dropout), name='dropout',
                            inputs=['forward', 'backward'])
        self.model.add_node(Dense(self.output_len, activation='softmax'),
                            name='softmax', input='dropout')
        self.model.add_output(name='output', input='softmax')

        # try using different optimizers and different optimizer configs
        self.model.compile(loss={'output': 'categorical_crossentropy'},
                           optimizer='rmsprop')

    def save_model(self):
        # save the model weight into a file
        self.model.save_weights('brnn_model_weights.h5')

    def plot_model(self):
        # plot the model, need the following packages:
        # pydot, graphviz, setuptools, pyparsing
        plot(self.model, to_file='brnn_model.png')

    @classmethod
    def sample(cls, prob, temperature=0.2):
        """
        softmax function for reinforcement learning
        """
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.argmax(np.random.multinomial(1, prob, 1))


def get_data():
    """
    retrieves data from a plain txt file and formats it
    using 1-of-k encoding
    """
    # read file and convert ids of each line into array of numbers
    with open("train_data", 'r') as f:
        sequence = [int(id_) for id_ in f]

    # vocabulary of the input sequence
    vocab = set(sequence)
    # add 0, representing 'no-log'
    vocab.add(0)
    # add another vocab, representing 'unknown-log'
    vocab.add(len(vocab))

    # number of template id types
    vocab_size = len(vocab)

    # length of one sentence
    sentence_length = 40
    # sample step per sentence
    step = 3

    # list of sentences
    sentences = []
    # list of the next id for each of the according sentence
    next_ids = []

    # creat batch data and next id sequences
    for i in range(0, sentence_length, step):
        sentences.append([0 for _ in range(0, sentence_length - i)] +
                          sequence[0: i])
        next_ids.append(sequence[i])
    for i in range(0, len(sequence) - sentence_length, step):
        sentences.append(sequence[i: i + sentence_length])
        next_ids.append(sequence[i + sentence_length])

    print "total # of sentences: %d" %len(sentences)

    # one-hot vector (all zeros except for a single one at
    # the exact postion of this id number)
    x = np.zeros((len(sentences), sentence_length, vocab_size), dtype=np.bool)
    # expected outputs for each sentence
    y = np.zeros((len(sentences), vocab_size), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, id_ in enumerate(sentence):
            # mark the each corresponding character in a sentence as 1
            x[i, t, id_] = 1
        # mark the corresponding character in expected output as 1
        y[i, next_ids[i]] = 1

    return sequence, sentence_length, vocab_size, x, y



print "Loading data..."
sequence, sentence_length, input_len, X_train, y_train = get_data()


# the size of each hidden layer
hidden_len = 512

# two layered LSTM 512 hidden nodes and a dropout rate of 0.2
brnn = SequenceAnalyzer(sentence_length, input_len, hidden_len, input_len)


print "Building Model..."
brnn.build_lstm(dropout=0.2)


print "Train..."
brnn.model.fit({'input': X_train, 'output': y_train}, validation_split=0.1,
               verbose=1, batch_size=128, nb_epoch=1)


# acc = accuracy(
#     y_test, np.round(np.array(model.predict({'input': X_test},
#                                             batch_size=batch_size)['output'])))
# print "Test accuracy: %.4f" %acc
