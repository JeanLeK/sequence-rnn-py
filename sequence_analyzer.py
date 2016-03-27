"""
This program analyze the integer sequence using Recurrent Neural Network (RNN)
(Uni-directional and Bi-directional) with Long Short-Term Memory (LSTM) and
Gated Recurrent Unit (GRU) based on the python library Keras.

"Keras is a minimalist, highly modular neural networks library, written in
 Python and capable of running on top of either TensorFlow or Theano."

                                                ---- Keras (http://keras.io/)

Uni-directional model is based on the Keras example - lstm_text_generation:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Bi-directional model is based on the Keras example - imdb_bidirectional_lstm.py:
https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

Author: Chang Liu (fluency03)
Data: 2016-03-27
"""

import sys
import random
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Graph
from keras.utils.visualize_util import plot


# random number generator with a fixed value for reproducibility
np.random.seed(1337)



class SequenceAnalyzer(object):
    """
    An integer sequence analyzer.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len,
                 return_sequence=True):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.return_sequence = return_sequence
        # model is defined at child class
        self.model = None

    def build_lstm(self, dropout=0.2):
        """
        Build model, defined at child class based on different model:
        Sequential or Graph.
        """
        pass

    def build_gru(self, dropout=0.2):
        """
        Build model, defined at child class based on different model:
        Sequential or Graph.
        """
        pass

    def save_model(self, filename):
        """
        Save the model weight into a hdf5 file
        """
        self.model.save_weights(filename)

    def plot_model(self, filename):
        """
        Plot the model, need the following packages:
        pydot, graphviz, setuptools, pyparsing
        """
        plot(self.model, to_file=filename)

    @classmethod
    def sample(cls, prob, temperature=0.2):
        """
        Softmax function for reinforcement learning
        """
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.argmax(np.random.multinomial(1, prob, 1))



class URNN(SequenceAnalyzer):
    """
    Uni-directional RNN model of the sequence analyzer. Sequential Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len,
                 return_sequence=True):
        super(URNN, self).__init__(sentence_length, input_len,
                                   hidden_len, output_len,
                                   return_sequence=return_sequence)
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

    def build_gru(self, dropout=0.2):
        """
        Stacked GRU with specified dropout rate, a model built with
        softmax activation, cross entropy loss and rmsprop optimizer
        """
        # 2 layer GRU with specified number of nodes in the hidden layer.
        self.model.add(GRU(self.hidden_len,
                           return_sequences=self.return_sequence,
                           input_shape=(self.sentence_length,
                                        self.input_len)))
        self.model.add(Dropout(dropout))

        self.model.add(GRU(self.hidden_len, return_sequences=False))
        self.model.add(Dropout(dropout))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def save_model(self):
        """
        Save the model weight into a hdf5 file
        """
        super(URNN, self).save_model('rnn_model_weights.h5')

    def plot_model(self):
        """
        Plot the model, need the following packages:
        pydot, graphviz, setuptools, pyparsing
        """
        super(URNN, self).plot_model('rnn_model.png')



class BRNN(SequenceAnalyzer):
    """
    Bi-directional RNN model of the sequence analyzer. Graph Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len,
                 return_sequence=True):
        super(BRNN, self).__init__(sentence_length, input_len,
                                   hidden_len, output_len,
                                   return_sequence=return_sequence)
        self.model = Graph()

    def build_lstm(self, dropout=0.2):
        """
        Bidirectional LSTM with specified dropout rate, a model built with
        softmax activation, cross entropy loss and rmsprop optimizer.
        Two RNN LSTMs stacked on top of each other.
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

    def build_gru(self, dropout=0.2):
        """
        Bidirectional GRU with specified dropout rate, a model built with
        softmax activation, cross entropy loss and rmsprop optimizer.
        Two RNN GRUs stacked on top of each other.
        """
        self.model.add_input(input_shape=(self.sentence_length, self.input_len),
                             name='input', dtype='float')
        self.model.add_node(GRU(self.hidden_len),
                            name='forward', input='input')
        self.model.add_node(GRU(self.hidden_len, go_backwards=True),
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
        """
        Save the model weight into a hdf5 file
        """
        super(BRNN, self).save_model('brnn_model_weights.h5')

    def plot_model(self):
        """
        Plot the model, need the following packages:
        pydot, graphviz, setuptools, pyparsing
        """
        super(BRNN, self).plot_model('brnn_model.png')



class History(Callback):
    """
    Record the loss and accuracy history
    """
    def on_train_begin(self, logs={}):
        # training loss and accuracy
        self.train_losses = []
        self.train_acc = []
        # validation loss and accuracy
        self.val_losses = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        # record training loss and accuracy
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        # record validation loss and accuracy
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))



def get_data():
    """
    Retrieves data from a plain txt file and formats it using one-hot vector
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



def print_losses(history):
    """
    Print the loss and accuracy
    """
    # print the losses and accuracy of training
    print "Training: "
    train_losses = history.train_losses
    train_acc = history.train_acc
    for l, a in zip(train_losses, train_acc):
        print "     Loss: %.4f, Accuracy: %.4f" %(l, a)

    # print the losses and accuracy of validation
    print "Validation: "
    val_losses = history.val_losses
    val_acc = history.val_acc
    for l, a in zip(val_losses, val_acc):
        print "     Loss: %.4f, Accuracy: %.4f" %(l, a)


def train(model='urnn'):
    """
    Trains the network and outputs the generated new sequence.

    Argument:
        model: Specify the model type, i.e.,
            urnn - Uni-directional RNN
            brnn - Bi-directional RNN
    """
    # get parameters and dimensions of the model
    print "Loading data..."
    sequence, sentence_length, input_len, x, y = get_data()

    # the size of each hidden layer
    hidden_len = 512

    # check model type: urnn or brnn
    if model == 'urnn':
        # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
        analyzer = URNN(sentence_length, input_len, hidden_len, input_len)
    elif model == 'brnn':
        # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
        # forward and backward
        analyzer = BRNN(sentence_length, input_len,
                        hidden_len, input_len)

    print "Building Model..."
    analyzer.build_lstm(dropout=0.2)

    # number of iterations
    nb_iterations = 40
    # train model and output generated sequence
    for iteration in range(1, nb_iterations+1):
        print ""
        print "------------------------ Start Training ------------------------"
        print "Iteration: ", iteration

        # history of losses and accuracy
        history = History()

        # saves the model weights after each epoch
        # if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       verbose=1, save_best_only=True)

        # train the model
        analyzer.model.fit(x, y, batch_size=128, nb_epoch=1,
                           validation_split=0.1, show_accuracy=True, verbose=1,
                           callbacks=[history, checkpointer])

        # start index of the seed, random number in range
        start_index = random.randint(0, len(sequence) - sentence_length - 1)

        # the Temperature option list
        t_list = [0.2, 0.5]

        # predict
        for T in t_list:
            print "------------Temperature: %.2f" %T
            sentence = sequence[start_index:start_index + sentence_length]
            # print sentence
            generated = sentence
            print "With seed: " + ' '.join(str(s) for s in sentence) + '\n'
            sys.stdout.write("Generated: " + ' '.join(str(g)
                                                      for g in generated))

            # generate 100 elements
            for _ in range(100):
                seed = np.zeros((1, sentence_length, input_len))
                # format input
                for t in range(0, sentence_length):
                    seed[0, t, sentence[t]] = 1

                # get predictions
                # verbose = 0, no logging
                predictions = analyzer.model.predict(seed, verbose=0)[0]
                # print "predictions length: %d" %len(predictions)
                next_id = analyzer.sample(predictions, T)
                # print predictions[next_id]
                # print next id
                sys.stdout.write(' ' + str(next_id))
                sys.stdout.flush()

                # use current output as input to predict the
                # next id in the sequence
                generated.append(next_id)
                sentence.pop(0)
                sentence.append(next_id)
            print "\n"

        # print the losses and accuracy
        print_losses(history)


if __name__ == '__main__':
    train()
