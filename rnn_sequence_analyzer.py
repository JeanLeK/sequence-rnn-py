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

from math import log
import glob
# import os
import sys
import csv
import time
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop # pylint: disable=W0611
from keras.utils.visualize_util import plot


# random number generator with a fixed value for reproducibility
np.random.seed(1337)


def override(f):
    """
    Override decorator.
    """
    return f


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

    def build(self, layer='LSTM', mapping='m2m', learning_rate=0.001,
              nb_layers=2, dropout=0.2):
        """
        Stacked RNN with specified dropout rate (default 0.2), built with
        softmax activation, cross entropy loss and rmsprop optimizer.

        Arguments:
            layer: {string}, the type of the layers in the RNN Model.
                'LSTM': LSTM layers
                'GRU': GRU layers
            mapping: {string}, input to output mapping.
                'o2o': one-to-one
                'm2m': many-to-many
            learning_rate: {float}, learning rate.
            nb_layers: {integer}, number of layers in total.
            dropout: {float}, dropout value.
        """
        print "Building Model..."
        print ("    layer = %d-%s , mapping = %s , learning rate = %.5f "
               "nb_layers = %d , dropout = %.2f"
               %(self.hidden_len, layer, mapping, learning_rate,
                 nb_layers, dropout))

        # check the layer type: LSTM or GRU
        if layer == 'LSTM':
            class LAYER(LSTM):
                """
                LAYER as LSTM.
                """
                pass
        elif layer == 'GRU':
            class LAYER(GRU):
                """
                LAYER as GRU.
                """
                pass

        # check whether return sequence for each of the layers
        return_sequences = []
        if mapping == 'o2o':
            # if mapping is one-to-one
            for nl in range(nb_layers):
                if nl == nb_layers-1:
                    return_sequences.append(False)
                else:
                    return_sequences.append(True)
        elif mapping == 'm2m':
            # if mapping is many-to-many
            for _ in range(nb_layers):
                return_sequences.append(True)

        # first layer RNN with specified number of nodes in the hidden layer.
        self.model.add(LAYER(self.hidden_len,
                             return_sequences=return_sequences[0],
                             input_shape=(self.sentence_length,
                                          self.input_len)))
        self.model.add(Dropout(dropout))

        # the following layers
        for nl in range(nb_layers-1):
            self.model.add(LAYER(self.hidden_len,
                                 return_sequences=return_sequences[nl+1]))
            self.model.add(Dropout(dropout))

        if mapping == 'o2o':
            # if mapping is one-to-one
            self.model.add(Dense(self.output_len))
        elif mapping == 'm2m':
            # if mapping is many-to-many
            self.model.add(TimeDistributed(Dense(self.output_len)))

        self.model.add(Activation('softmax'))

        rms = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=rms,
                           metrics=['accuracy'])

    def save_model(self, filename, overwrite=False):
        """
        Save the model weight into a hdf5 file.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be saved.
            overwrite: {bool}, overwrite existing file.
        """
        print "Save Weights %s ..." %filename
        self.model.save_weights(filename, overwrite=overwrite)

    def load_model(self, filename):
        """
        Load the model weight into a hdf5 file.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be loaded.
        """
        print "Load Weights %s ..." %filename
        self.model.load_weights(filename)

    def plot_model(self, filename='rnn_model.png'):
        """
        Plot model.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be plotted.
        """
        print "Plot Model %s ..." %filename
        plot(self.model, to_file=filename)


class History(Callback):
    """
    Record the loss and accuracy history.
    """
    @override
    def on_train_begin(self, logs={}): # pylint: disable=W0102
        """
        A method starting at the begining of the training.

        Arguments:
            logs: {dictionary}, recording the training and validation
                losses and accuracy of every epoch.
        """
        # training loss and accuracy
        self.train_losses = []
        self.train_acc = []
        # validation loss and accuracy
        self.val_losses = []
        self.val_acc = []

    @override
    def on_epoch_end(self, epoch, logs={}): # pylint: disable=W0102
        """
        A method starting at the begining of the training.

        Arguments:
            epoch: {integer}, the current epoch.
            logs: {dictionary}, recording the training and validation
                losses and accuracy of every epoch.
        """
        # record training loss and accuracy
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        # record validation loss and accuracy
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # continutously save the train_loss, train_acc, val_loss, val_acc
        # into a csv file with 4 columns respeactively
        csv_name = 'history.csv'
        with open(csv_name, 'a') as csvfile:
            his_writer = csv.writer(csvfile)
            print "\n    Save loss and accuracy into %s" %csv_name
            his_writer.writerow((logs.get('loss'), logs.get('acc'),
                                 logs.get('val_loss'), logs.get('val_acc')))


def sample(prob, temperature=0.2):
    """
    Softmax function for reinforcement learning.

    Arguments:
        prob: {list}, a list of probabilities of each of the classes.
        temperature: {float}, Softmax temperature.
    Returns:
        {integer}, the most possible sample.
    """
    prob = np.log(prob) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    return np.argmax(np.random.multinomial(1, prob, 1))


def get_sequence(filepath):
    """
    Get the original sequence from file.

    Arguments:
        filename: {string}, the name/path of input log sequence file.
    Returns:
        {list}, the log sequence.
        {integer}, the size of vocabulary.
    """
    # read file and convert ids of each line into array of numbers
    seqfiles = glob.glob(filepath)
    sequence = []

    for seqfile in seqfiles:
        with open(seqfile, 'r') as f:
            one_sequence = [int(id_) for id_ in f]
            print "        %s, sequence length: %d" %(seqfile,
                                                      len(one_sequence))
            sequence.extend(one_sequence)

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = max(sequence) + 2

    return sequence, vocab_size


def get_data(sequence, vocab_size, mapping='m2m', sentence_length=40, step=3,
             random_offset=True):
    """
    Retrieves data from a plain txt file and formats it using one-hot vector.

    Arguments:
        sequence: {lsit}, the original input sequence
        vocab_size: {integer}, the number of unique id classes
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        random_offset: {bool}, the offset is random between step or is 0.
    Returns:
        {np.array}, training input data X
        {np.array}, training target data y
    """
    X_sentences = []
    y_sentences = []
    next_ids = []

    offset = np.random.randint(0, step) if random_offset else 0

    # creat batch data and next sentences
    for i in range(offset, len(sequence) - sentence_length, step):
        X_sentences.append(sequence[i : i + sentence_length])
        if mapping == 'o2o':
            # if mapping is one-to-one
            next_ids.append(sequence[i + sentence_length])
        elif mapping == 'm2m':
            # if mapping is many-to-many
            y_sentences.append(sequence[i + 1 : i + sentence_length + 1])

    # number of sampes
    nb_samples = len(X_sentences)
    # print "total # of sentences: %d" %nb_samples

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

    return X_train, y_train


def predict(sequence, input_len, analyzer, nb_predictions=80,
            mapping='m2m', sentence_length=40):
    """
    Predict the next sequences using existing model and weights given some seed.

    Arguments:
        sequence: {lsit}, the original input sequence
        input_len: {integer}, the number of unique id classes
        analyzer: {SequenceAnalyzer}, the sequence analyzer
        nb_predictions: {integer}, number of predictions after giving the seed
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each sentence.
    """
    # generate elements
    for _ in range(nb_predictions):
        # start index of the seed, random number in range
        start_index = np.random.randint(0, len(sequence) - sentence_length - 1)
        # seed sentence
        sentence = sequence[start_index : start_index + sentence_length]

        # Y_true
        y_true = sequence[start_index + 1 : start_index + sentence_length + 1]
        print "X:      " + ' '.join(str(s).ljust(4) for s in sentence)

        seed = np.zeros((1, sentence_length, input_len))
        # format input
        for t in range(0, sentence_length):
            seed[0, t, sentence[t]] = 1

        # get predictions
        # verbose = 0, no logging
        predictions = analyzer.model.predict(seed, verbose=0)[0]

        # y_predicted
        if mapping == 'o2o':
            next_id = np.argmax(predictions)
            sys.stdout.write(' ' + str(next_id))
            sys.stdout.flush()
        elif mapping == 'm2m':
            next_sentence = []
            for pred in predictions:
                next_sentence.append(np.argmax(pred))
            print "y_pred: " + ' '.join(str(id_).ljust(4)
                                        for id_ in next_sentence)
            # next_id = np.argmax(predictions[-1])

        # y_true
        print "y_true: " + ' '.join(str(s).ljust(4) for s in y_true)

        print "\n"


def train(analyzer, train_sequence, val_sequence, input_len,
          batch_size=128, nb_epoch=50, nb_iterations=4,
          sentence_length=40, step=40, mapping='m2m'):
    """
    Trains the network.

    Arguments:
        analyzer: {SequenceAnalyzer}.
        train_sequence: {list}, training sequence.
        val_sequence: {list}, validation sequence.
        input_len: {integer}, the number of classes, i.e., the input length of
            neural network.
        batch_size: {interger}, the number of sentences per batch.
        nb_epoch: {integer}, number of epoches per iteration.
        nb_iterations: {integer}, number of iterations.
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
    """
    for iteration in range(1, nb_iterations+1):
        # create training data, randomize the offset between steps
        X_train, y_train = get_data(train_sequence, input_len, mapping=mapping,
                                    sentence_length=sentence_length, step=step,
                                    random_offset=False)
        X_val, y_val = get_data(val_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=False)
        print ""
        print "------------------------ Start Training ------------------------"
        print "Iteration: ", iteration
        print "Number of epoch per iteration: ", nb_epoch

        # history of losses and accuracy
        history = History()

        # saves the model weights after each epoch
        # if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       verbose=1, save_best_only=True)

        # train the model
        analyzer.model.fit(X_train, y_train,
                           batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                           callbacks=[history, checkpointer],
                           validation_data=(X_val, y_val))

        analyzer.save_model("weights-after-iteration.hdf5", overwrite=True)


def detect(sequence, input_len, analyzer, mapping='m2m', sentence_length=40):
    """
    Scan the given sequence for detecting anormalies.

    Arguments:
        sequence: {lsit}, the original input sequence
        input_len: {integer}, the number of unique id classes
        analyzer: {SequenceAnalyzer}, the sequence analyzer
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each sentence.
    """
    # sequence length
    length = len(sequence)

    # predicted probabilities for each id
    # we assume the first sentence_length ids are true
    prob = [1] * sentence_length + [0] * (length - sentence_length)
    log_prob = [0] * length

    start_time = time.time()
    try:
        # generate elements
        for start_index in xrange(length - sentence_length):
            # seed sentence
            X = sequence[start_index : start_index + sentence_length]
            # print "X:      " + ' '.join(str(s).ljust(4) for s in sentence)

            # Y_true
            # y_true = sequence[start_index + 1 : start_index + sentence_length + 1]
            # print "y_true: " + ' '.join(str(s).ljust(4) for s in y_true)
            y_next_true = sequence[start_index + sentence_length]

            seed = np.zeros((1, sentence_length, input_len))
            # format input
            for t in range(0, sentence_length):
                seed[0, t, X[t]] = 1

            # get predictionsverbose = 0, no logging
            predictions = analyzer.model.predict(seed, verbose=0)[0]

            # y_predicted
            y_next_pred = 0
            next_prob = 0
            if mapping == 'o2o':
                next_prob = predictions[y_next_true]
                # prob[start_index + sentence_length] = next_prob
                y_next_pred = np.argmax(predictions)
            elif mapping == 'm2m':
                # next_sentence = []
                # for pred in predictions:
                #     next_sentence.append(np.argmax(pred))
                # y_next_pred = next_sentence[-1]
                # print "y_pred: " + ' '.join(str(id_).ljust(4)
                #                             for id_ in next_sentence)
                y_next_pred = np.argmax(predictions[-1])
                next_prob = predictions[-1][y_next_true]

            next_prob = 1.0 if y_next_pred == y_next_true else next_prob
            prob[start_index + sentence_length] = next_prob
            log_prob[start_index + sentence_length] = -log(next_prob)

            print start_index, next_prob
    except KeyboardInterrupt:
        pass
        # print "    |-Write the clusters into %s ..." %self.cluster_file
        # with open('prob_new.txt', 'w') as prob_file:
        #     for p in prob:
        #         prob_file.write(str(p) + '\n')
        #
        # plt.plot(prob, 'r*')
        # plt.xlim(0, 1000)
        # plt.ylim(0, 1)
        # plt.savefig("prob1.png")
        # plt.clf()
        # plt.cla()

    with open('prob_new.txt', 'w') as prob_file:
        for p in prob:
            prob_file.write(str(p) + '\n')

    with open('log_prob_new.txt', 'w') as prob_file:
        for log_p in log_prob:
            prob_file.write(str(log_p) + '\n')

    stop_time = time.time()
    print "--- %s seconds ---\n" % (stop_time - start_time)

    return prob


def run(hidden_len=512, batch_size=128, nb_epoch=50, nb_iterations=4,
        learning_rate=0.001, nb_predictions=20, mapping='o2o',
        sentence_length=40, step=40, mode='evaluate'):
    """
    Train, evaluate, or predict.

    Arguments:
        hidden_len: {integer}, the size of a hidden layer.
        batch_size: {interger}, the number of sentences per batch.
        nb_epoch: {interger}, number of epoches per iteration.
        nb_iterations: {integer}, number of iterations.
        learning_rate: {float}, learning rate.
        nb_predictions: {integer}, number of the ids predicted.
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        mode: {string}, th running mode of this programm
            'train': train and predict
            'predict': only predict by loading existing model weights
            'evaluate': evaluate the model in evaluation data set
            'detect': detect a new log sequence for the probabilities
    """
    # get parameters and dimensions of the model
    print "Loading training data..."
    train_sequence, input_len1 = get_sequence("./train_data/*")
    print "Loading validation data..."
    val_sequence, input_len2 = get_sequence("./validation_data/*")
    input_len = max(input_len1, input_len2)

    print "Training sequence length: %d" %len(train_sequence)
    print "Validation sequence length: %d" %len(val_sequence)
    print "#classes: %d\n" %input_len

    # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
    rnn = SequenceAnalyzer(sentence_length, input_len, hidden_len, input_len)

    # build model
    rnn.build(layer='LSTM', mapping=mapping, learning_rate=learning_rate,
              nb_layers=3, dropout=0.5)

    # plot model
    # rnn.plot_model()

    # load the previous model weights
    rnn.load_model("weights-after-iteration-l1.hdf5")

    if mode == 'predict':
        print "Predict..."
        predict(val_sequence, input_len, rnn, nb_predictions=nb_predictions,
                mapping=mapping, sentence_length=sentence_length)
    elif mode == 'evaluate':
        print "Evaluate..."
        print "Metrics: " + ', '.join(rnn.model.metrics_names)
        X_val, y_val = get_data(val_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=False)
        results = rnn.model.evaluate(X_val, y_val, #pylint: disable=W0612
                                     batch_size=batch_size,
                                     verbose=1)
        print "Loss: ", results[0]
        print "Accuracy: ", results[1]
    elif mode == 'train':
        print "Train..."
        try:
            train(rnn, train_sequence, val_sequence, input_len,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  nb_iterations=nb_iterations,
                  sentence_length=sentence_length,
                  step=step, mapping=mapping)
        except KeyboardInterrupt:
            rnn.save_model("weights-stop.hdf5", overwrite=True)
    elif mode == 'detect':
        print "Detect..."
        detect(val_sequence, input_len, rnn, mapping=mapping,
               sentence_length=sentence_length)
    else:
        print "The mode = %s is not correct!!!" %mode

    return mode


if __name__ == '__main__':
    run()
