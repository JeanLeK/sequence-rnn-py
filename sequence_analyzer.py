"""
This program analyze the integer sequence using Recurrent Neural Network (RNN)
with Long Short-Term Memory (LSTM) based on the python library Keras.

"Keras is a minimalist, highly modular neural networks library, written in
Python and capable of running on top of either TensorFlow or Theano.""
                                                ---- Keras (http://keras.io/)



Author: Chang Liu (fluency03)
Data: 2016-03-17
"""

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
import sys

# from IPython import embed


class IntSequenceAnalyzer(object):
    """
    A integer sequence analyzer.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len,
                 return_sequence=True):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.return_sequence = return_sequence
        self.model = Sequential()

    def build_lstm(self, dropout=0.2):
        """
        Stacked LSTM with specified dropout rate, a model built with
        softmax activation, cross entropy loss and rmsprop optimizer
        """
        # 2 layer LSTM with specified number of nodes in the hidden layer.
        self.model.add(LSTM(self.hidden_len,
                            return_sequences=self.return_sequence,
                            input_shape=(self.sentence_length,
                                         self.input_len)))
        self.model.add(Dropout(dropout))

        self.model.add(LSTM(self.hidden_len, return_sequences=False))
        self.model.add(Dropout(dropout))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    @classmethod
    def sample(cls, prob, temperature=1.0):
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
    with open("sequence", 'r') as f:
        sequence = [int(id_) for id_ in f]

    # vocabulary of the input sequence
    vocab = set(sequence)
    # add 0, representing
    vocab.add(0)

    # number of template id types
    vocab_size = len(vocab)

    # length of one sentence
    sentence_length = 20
    # sample step per sentence
    step = 3

    # list of sentences
    sentences = []
    # list of the next id for each of the according sentence
    next_ids = []

    # creat batch data and next id sequences
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


def train():
    """
    Trains the network and outputs the generated text.
    Trains using batch size of 100, 60 epochs total.
    """
    sequence, sentence_length, input_len, x, y = get_data()

    hidden_len = 512
    # two layered LSTM 512 hidden nodes and a dropout rate of 0.5
    lstm = IntSequenceAnalyzer(sentence_length,
                               input_len, hidden_len, input_len)
    print "Building Model..."
    # IPython.embed()
    lstm.build_lstm(dropout=0.2)

    # train model and output generated text
    for iteration in range(1, 41):
        print ""
        print "------------------------ Start Training ------------------------"
        print "Iteration: ", iteration
        lstm.model.fit(x, y, batch_size=128, nb_epoch=1)

        start_index = random.randint(0, len(sequence) - sentence_length - 1)
        for T in [0.2, 0.5, 1.0, 1.2]:
            print "------------Temperature: %.2f" %T
            sentence = sequence[start_index:start_index + sentence_length]
            # print sentence
            generated = sentence
            print "With seed: " + ' '.join(str(s) for s in sentence) + '\n'
            sys.stdout.write("Generated: " + ' '.join(str(g)
                                                      for g in generated))

            # generate 400 chars
            for _ in range(100):
                seed = np.zeros((1, sentence_length, input_len))
                # format input
                for t in range(0, sentence_length):
                    seed[0, t, sentence[t]] = 1

                # get predictions
                # verbose = 0, no logging
                predictions = lstm.model.predict(seed, verbose=0)[0]
                # print "predictions length: %d" %len(predictions)
                next_id = lstm.sample(predictions, T)
                # print predictions[next_id]
                # print next id
                sys.stdout.write(' ' + str(next_id))
                sys.stdout.flush()

                # use current output as input to predict the
                # next id in the sequence
                generated.append(next_id)
                sentence.pop(0)
                sentence.append(next_id)

            print ""


if __name__ == '__main__':
    train()
