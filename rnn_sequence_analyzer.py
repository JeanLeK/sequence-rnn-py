"""
This program analyze the integer sequence using Uni-diractional Recurrent Neural
Network (RNN) with Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
based on the python library Keras.

"Keras is a minimalist, highly modular neural networks library, written in
 Python and capable of running on top of either TensorFlow or Theano."
                                                ---- Keras (http://keras.io/)

It is based on this Keras example - lstm_text_generation:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Author: Chang Liu (fluency03)
Data: 2016-03-17
"""

import sys
import random
import csv
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.core import Activation, Dense, TimeDistributedDense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot


# random number generator with a fixed value for reproducibility
np.random.seed(1337)


class SequenceAnalyzer(object):
    """
    Sequence analyzer based on RNN Sequential Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.model = Sequential()

    def build_lstm(self, mapping='o2o', nb_layers=2, dropout=0.2):
        """
        Stacked LSTM with specified dropout rate (default 0.2), built with
        softmax activation, cross entropy loss and rmsprop optimizer.

        Arguments:
            mapping: string, input to output mapping
                o2o: one-to-one
                m2m: many-to-many
            nb_layers: integer, number of layers in total
            dropout: float, dropout value
        """
        print "Building Model..."

        # check whether the last layer return sequences
        if mapping == 'o2o':
            # if mapping is one-to-one
            return_sequences = False
        elif mapping == 'm2m':
            # if mapping is many-to-many
            return_sequences = True

        # 2 layer LSTM with specified number of nodes in the hidden layer.
        self.model.add(LSTM(self.hidden_len, return_sequences=True,
                            input_shape=(self.sentence_length,
                                         self.input_len)))
        self.model.add(Dropout(dropout))

        for nl in range(nb_layers-1):
            # check whether return sequences
            if nl != nb_layers-2:
                return_sequences_ = True
            else:
                return_sequences_ = return_sequences
            # build hidden layers
            self.model.add(LSTM(self.hidden_len,
                                return_sequences=return_sequences_))
            self.model.add(Dropout(dropout))

        if mapping == 'o2o':
            # if mapping is one-to-one
            self.model.add(Dense(self.output_len))
        elif mapping == 'm2m':
            # if mapping is many-to-many
            self.model.add(TimeDistributedDense(self.output_len))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def build_gru(self, mapping='o2o', nb_layers=2, dropout=0.2):
        """
        Stacked GRU with specified dropout rate (default 0.2), built with
        softmax activation, cross entropy loss and rmsprop optimizer.

        Arguments:
            mapping: string, input to output mapping
                o2o: one-to-one
                m2m: many-to-many
            nb_layers: integer, number of layers in total
            dropout: float, dropout value
        """
        print "Building Model..."

        # check whether the last layer return sequences
        if mapping == 'o2o':
            # if mapping is one-to-one
            return_sequences = False
        elif mapping == 'm2m':
            # if mapping is many-to-many
            return_sequences = True

        # 2 layer GRU with specified number of nodes in the hidden layer.
        self.model.add(GRU(self.hidden_len, return_sequences=True,
                           input_shape=(self.sentence_length,
                                        self.input_len)))
        self.model.add(Dropout(dropout))

        for nl in range(nb_layers-1):
            # check whether return sequences
            if nl != nb_layers-2:
                return_sequences_ = True
            else:
                return_sequences_ = return_sequences
            # build hidden layers
            self.model.add(GRU(self.hidden_len,
                               return_sequences=return_sequences_))
            self.model.add(Dropout(dropout))

        if mapping == 'o2o':
            # if mapping is one-to-one
            self.model.add(Dense(self.output_len))
        elif mapping == 'm2m':
            # if mapping is many-to-many
            self.model.add(TimeDistributedDense(self.output_len))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def save_model(self, filename):
        """
        Save the model weight into a hdf5 file.
        """
        print "Save Weights..."
        self.model.save_weights(filename)

    def load_model(self, filename):
        """
        Load the model weight into a hdf5 file.
        """
        print "Load Weights..."
        self.model.load_weights(filename)

    def plot_model(self):
        """
        Plot model.
        """
        print "Plot Model..."
        plot(self.model, to_file='rnn_model.png')

    @classmethod
    def sample(cls, prob, temperature=0.2):
        """
        Softmax function for reinforcement learning.
        """
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.argmax(np.random.multinomial(1, prob, 1))



class History(Callback):
    """
    Record the loss and accuracy history.
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


def get_data(mapping='o2o', sentence_length=40, step=3):
    """
    Retrieves data from a plain txt file and formats it using one-hot vector.
    """
    # read file and convert ids of each line into array of numbers
    with open("/home/cliu/Documents/SC-1/sequence", 'r') as f:
        sequence = [int(id_) for id_ in f]

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = max(sequence) + 2

    X_sentences = []
    y_sentences = []
    next_ids = []

    # creat batch data and next sentences
    for i in range(0, len(sequence) - sentence_length, step):
        X_sentences.append(sequence[i : i + sentence_length])
        if mapping == 'o2o':
            # if mapping is one-to-one
            next_ids.append(sequence[i + sentence_length])
        elif mapping == 'm2m':
            # if mapping is many-to-many
            y_sentences.append(sequence[i + 1 : i + sentence_length + 1])

    # number of sampes
    nb_samples = len(X_sentences)
    print "total # of sentences: %d" %nb_samples

    # one-hot vector (all zeros except for a single one at
    # the exact postion of this id number)
    X_train = np.zeros((nb_samples, sentence_length, vocab_size), dtype=np.bool)
    # expected outputs for each sentence
    if mapping == 'o2o':
        # if mapping is one-to-one
        y_train = np.zeros((nb_samples, vocab_size), dtype=np.bool)
    elif mapping == 'm2m':
        # if mapping is many-to-many
        y_train = np.zeros((nb_samples, sentence_length, vocab_size),
                           dtype=np.bool)

    for i, x_sentence in enumerate(X_sentences):
        for t, id_ in enumerate(x_sentence):
            # mark the each corresponding character in a sentence as 1
            X_train[i, t, id_] = 1
            # if mapping is many-to-many
            if mapping == 'm2m':
                y_train[i, t, y_sentences[i][t]] = 1
        # if mapping is one-to-one
        # mark the corresponding character in expected output as 1
        if mapping == 'o2o':
            y_train[i, next_ids[i]] = 1

    return sequence, sentence_length, vocab_size, X_train, y_train



def print_save_losses(history):
    """
    Print the loss and accuracy, and continuously save them into a csv file
    """
    # print the losses and accuracy of training
    print "Training: "
    train_losses = history.train_losses
    train_acc = history.train_acc
    for l, a in zip(train_losses, train_acc):
        print "     Loss: %.4f , Accuracy: %.4f" %(l, a)

    # print the losses and accuracy of validation
    print "Validation: "
    val_losses = history.val_losses
    val_acc = history.val_acc
    for l, a in zip(val_losses, val_acc):
        print "     Loss: %.4f , Accuracy: %.4f" %(l, a)

    # continutously save the train_losses, train_acc, val_losses, val_acc
    # into a csv file with 4 columns respeactively
    rows = zip(train_losses, train_acc, val_losses, val_acc)
    with open('history.csv', 'a') as csvfile:
        his_writer = csv.writer(csvfile)
        for row in rows:
            his_writer.writerow(row)



def train(hidden_len=512, batch_size=128, nb_epoch=1, validation_split=0.1,
          show_accuracy=True, nb_iterations=40, nb_predictions=100,
          mapping='o2o'):
    """
    Trains the network and outputs the generated new sequence.

    Arguments:
        hidden_len: integer, the size of a hidden layer.
        batch_size: interger, the number of sentences per batch.
        nb_epoch: interger, number of epoches per iteration.
        validation_split: float (0 ~ 1), percentage of validation data
            among training data.
        show_accuracy: boolean, show accuracy during training.
        nb_iterations: integer, number of iterations.
        nb_predictions: integer, number of the ids predicted.
        mapping: input to output mapping
            o2o: one-to-one
            m2m: many-to-many
    """
    # get parameters and dimensions of the model
    print "Loading data..."
    sequence, sentence_length, input_len, X_train, y_train = get_data(
        mapping=mapping, sentence_length=40, step=3)

    # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
    rnn = SequenceAnalyzer(sentence_length, input_len, hidden_len, input_len)

    # build model
    rnn.build_lstm(mapping=mapping)

    # load the previous model weights
    rnn.load_model("weights4.hdf5")

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
        rnn.model.fit(X_train, y_train,
                      batch_size=batch_size, nb_epoch=nb_epoch,
                      validation_split=validation_split,
                      show_accuracy=show_accuracy,
                      verbose=1,
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
            for _ in range(nb_predictions):
                seed = np.zeros((1, sentence_length, input_len))
                # format input
                for t in range(0, sentence_length):
                    seed[0, t, sentence[t]] = 1

                # get predictions
                # verbose = 0, no logging
                predictions = rnn.model.predict(seed, verbose=0)[0]
                # print "predictions length: %d" %len(predictions)
                next_id = rnn.sample(predictions, T)
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
        print_save_losses(history)


if __name__ == '__main__':
    train(nb_iterations=30)
