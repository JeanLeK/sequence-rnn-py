"""
This program analyze the integer sequence using (Uni-directional and
Bi-directional) Recurrent Neural Network (RNN) with Long Short-Term Memory
(LSTM) and Gated Recurrent Unit (GRU) based on the python library Keras.

Input data is Generator and the training is by calling model.fit_generator().

"Keras is a minimalist, highly modular neural networks library, written in
 Python and capable of running on top of either TensorFlow or Theano."
                                                ---- Keras (http://keras.io/)

Uni-directional model is based on the Keras example - lstm_text_generation:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Bi-directional model is based on the Keras example - imdb_bidirectional_lstm.py:
https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

Author: Chang Liu (fluency03)
Data: 2016-04-03
"""

import sys
import csv
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.core import Activation, Dense, TimeDistributedDense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Graph
from keras.optimizers import RMSprop
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
    Sequence analyzer based on RNN.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        # model is defined at child class
        self.model = None

    def build(self, layer, mapping, nb_layers, dropout):
        """
        Build model.
        """
        pass

    def save_model(self, filename):
        """
        Save the model weight into a hdf5 file.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be saved.
        """
        print "Save Weights..."
        self.model.save_weights(filename)

    def load_model(self, filename):
        """
        Load the model weight into a hdf5 file.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be loaded.
        """
        print "Load Weights..."
        self.model.load_weights(filename)

    def plot_model(self, filename):
        """
        Plot model.

        Arguments:
            filename: {string}, the name/path to the file
                to which the model graphic is plotted.
        """
        print "Plot Model..."
        plot(self.model, to_file=filename)


class URNN(SequenceAnalyzer):
    """
    Uni-directional RNN model of the sequence analyzer. Sequential Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        super(URNN, self).__init__(sentence_length,
                                   input_len, hidden_len, output_len,
                                   return_sequence=True)
        self.model = Sequential()

    @override
    def build(self, layer='LSTM', mapping='o2o', nb_layers=2, dropout=0.2):
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
            nb_layers: {integer}, number of layers in total.
            dropout: {float}, dropout value.
        """
        print "Building Model..."

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
            self.model.add(TimeDistributedDense(self.output_len))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


class BRNN(SequenceAnalyzer):
    """
    Bi-directional RNN model of the sequence analyzer. Graph Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        super(BRNN, self).__init__(sentence_length,
                                   input_len, hidden_len, output_len)
        self.model = Graph()

    @override
    def build(self, layer='LSTM', mapping='o2o', nb_layers=2, dropout=0.2):
        """
        Bidirectional RNN with specified dropout rate (default 0.2), built with
        softmax activation, cross entropy loss and rmsprop optimizer.

        Arguments:
            layer: {string}, the type of the layers in the RNN Model.
                'LSTM': LSTM layers
                'GRU': GRU layers
            mapping: {string}, input to output mapping.
                'o2o': one-to-one
                'm2m': many-to-many
            nb_layers: {integer}, number of layers in total.
            dropout: {float}, dropout value.
        """
        print "Building Model..."

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

        # add input
        self.model.add_input(input_shape=(self.sentence_length, self.input_len),
                             name='input', dtype='float')

        # first Bi-directional LSTM layer
        self.model.add_node(LAYER(self.hidden_len,
                                  return_sequences=return_sequences[0]),
                            name='forward1', input='input')
        self.model.add_node(Dropout(dropout),
                            name='forward_dropout1', input='forward1')
        self.model.add_node(LAYER(self.hidden_len,
                                  return_sequences=return_sequences[0],
                                  go_backwards=True),
                            name='backward1', input='input')
        self.model.add_node(Dropout(dropout),
                            name='backward_dropout1', input='backward1')

        # following Bi-directional layers
        for nl in range(nb_layers-1):
            self.model.add_node(LAYER(self.hidden_len,
                                      return_sequences=return_sequences[nl+1]),
                                name='forward' + str(nl+2),
                                input='forward_dropout' + str(nl+1))
            self.model.add_node(Dropout(dropout),
                                name='forward_dropout' + str(nl+2),
                                input='forward' + str(nl+2))
            self.model.add_node(LAYER(self.hidden_len,
                                      return_sequences=return_sequences[nl+1],
                                      go_backwards=True),
                                name='backward' + str(nl+2),
                                input='backward_dropout' + str(nl+1))
            self.model.add_node(Dropout(dropout),
                                name='backward_dropout' + str(nl+2),
                                input='backward' + str(nl+2))

        # self.model.add_node(Dropout(dropout), name='dropout',
                            # inputs=['forward', 'backward'])
        if mapping == 'o2o':
            self.model.add_node(Dense(self.output_len, activation='softmax'),
                                name='softmax',
                                inputs=['forward_dropout' + str(nb_layers),
                                        'backward_dropout' + str(nb_layers)])
        elif mapping == 'm2m':
            self.model.add_node(TimeDistributedDense(self.output_len,
                                                     activation='softmax'),
                                name='softmax',
                                inputs=['forward_dropout' + str(nb_layers),
                                        'backward_dropout' + str(nb_layers)])

        # add ouput
        self.model.add_output(name='output', input='softmax')

        # try using different optimizers and different optimizer configs
        self.model.compile(loss={'output': 'categorical_crossentropy'},
                           optimizer='rmsprop')


class History(Callback):
    """
    Record the loss and accuracy history.
    """
    @override
    def on_train_begin(self, logs={}):
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
    def on_epoch_end(self, epoch, logs={}):
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


def get_sequence(filename):
    """
    Get the original sequence from file.

    Arguments:
        filename: {string}, the name/path of input log sequence file.
    Returns:
        {list}, the log sequence.
        {integer}, the size of vocabulary.
    """
    # read file and convert ids of each line into array of numbers
    with open(filename, 'r') as f:
        sequence = [int(id_) for id_ in f]

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = max(sequence) + 2

    return sequence, vocab_size


def data_generator(sequence, vocab_size, mapping='o2o', sentence_length=40,
                   step=3, offset=0, batch_size=128):
    """
    Retrieves data from a plain txt file and formats it using one-hot vector.
    This method returns a data generator yeilding a batch of (X_train, y_train)
    every time being called.

    Arguments:
        sequence: {lsit}, the original input sequence
        vocab_size: {integer}, the number of unique id classes
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        offset: {integer}, the offset of starting point of sampling.
        batch_size: {integer}, the number of sample per batch.
    Yields:
        {np.array}, training input data X
        {np.array}, training target data y
    """
    # the number of current sample
    sample_count = 0

    # one-hot vector (all zeros except for a single one at
    # the exact postion of this id number)
    X_train = np.zeros((batch_size, sentence_length, vocab_size),
                       dtype=np.bool)
    # expected outputs for each sentence
    if mapping == 'o2o':
        # if mapping is one-to-one
        y_train = np.zeros((batch_size, vocab_size), dtype=np.bool)
    elif mapping == 'm2m':
        # if mapping is many-to-many
        y_train = np.zeros((batch_size, sentence_length, vocab_size),
                           dtype=np.bool)

    # continuousy creat batch data and next sentences
    while True:
        for i in range(offset, len(sequence) - offset - sentence_length, step):
            # index of a this sample in this batch
            batch_index = sample_count % batch_size

            # re-initialzing the batch
            if batch_index == 0:
                X_train.fill(0)
                y_train.fill(0)

            # current sample and target outputs
            X_sentence = []
            y_sentence = []
            next_id = []

            X_sentence = sequence[i : i + sentence_length]
            if mapping == 'o2o':
                # if mapping is one-to-one
                next_id = sequence[i + sentence_length]
            elif mapping == 'm2m':
                # if mapping is many-to-many
                y_sentence = sequence[i + 1 : i + sentence_length + 1]

            for t, id_ in enumerate(X_sentence):
                # mark the each corresponding character in a sentence as 1
                X_train[batch_index, t, id_] = 1
                # if mapping is many-to-many
                if mapping == 'm2m':
                    y_train[batch_index, t, y_sentence[t]] = 1
            # if mapping is one-to-one
            # mark the corresponding character in expected output as 1
            if mapping == 'o2o':
                y_train[batch_index, next_id] = 1

            # sample count plus 1
            sample_count += 1

            if batch_index == batch_size-1:
                yield X_train, y_train


def print_save_losses(history):
    """
    Print the loss and accuracy, and continuously save them into a csv file.

    Arguments:
        history: {History}, the callbacks recording losses and accuracy.
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
    # start index of the seed, random number in range
    start_index = np.random.randint(0, len(sequence) - sentence_length - 1)


    sentence = sequence[start_index:start_index + sentence_length]
    # print sentence
    generated = sentence
    print "With seed: " + ' '.join(str(s) for s in sentence) + '\n'
    sys.stdout.write("Generated: " + ' '.join(str(g)
                                              for g in generated) + '\n')

    # generate elements
    for _ in range(nb_predictions):
        seed = np.zeros((1, sentence_length, input_len))
        # format input
        for t in range(0, sentence_length):
            seed[0, t, sentence[t]] = 1

        # get predictions
        # verbose = 0, no logging
        predictions = analyzer.model.predict(seed, verbose=0)[0]

        # print "predictions length: %d" %len(predictions)
        # print predictions.shape
        if mapping == 'o2o':
            next_id = np.argmax(predictions)
            sys.stdout.write(' ' + str(next_id))
            sys.stdout.flush()
        elif mapping == 'm2m':
            next_sentence = []
            for pred in predictions:
                next_sentence.append(np.argmax(pred))
            print ' '.join(str(id_) for id_ in next_sentence)
            next_id = np.argmax(predictions[-1])

        # use current output as input to predict the
        # next id in the sequence
        generated.append(next_id)
        sentence.pop(0)
        sentence.append(next_id)

    print "\n"


def train(model='urnn', hidden_len=512, batch_size=128, nb_batch=40, nb_epoch=1,
          show_accuracy=True, nb_iterations=40, nb_predictions=100,
          mapping='o2o', sentence_length=40, step=3, mode='train'):
    """
    Trains the network and outputs the generated new sequence.

    Arguments:
        hidden_len: {integer}, the size of a hidden layer.
        batch_size: {interger}, the number of sentences per batch.
        nb_batch: {integer}, number of batches to be trained durign each epoch.
        nb_epoch: {interger}, number of epoches per iteration.
        validation_split: {float} (0 ~ 1), percentage of validation data
            among training data.
        show_accuracy: {boolean}, show accuracy during training.
        nb_iterations: {integer}, number of iterations.
        nb_predictions: {integer}, number of the ids predicted.
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        mode: {string}, th running mode of this programm
            'train': train and predict
            'predict': only predict by loading existing model weights
    """
    # get parameters and dimensions of the model
    print "Loading data..."
    sequence, input_len = get_sequence("/home/cliu/Documents/SC-1/sequence")

    # check model type: urnn or brnn
    if model == 'urnn':
        # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
        analyzer = URNN(sentence_length, input_len, hidden_len, input_len)
    elif model == 'brnn':
        # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
        # forward and backward
        analyzer = BRNN(sentence_length, input_len,
                        hidden_len, input_len)

    # build model
    analyzer.build(layer='LSTM', mapping=mapping)

    # load the previous model weights
    # analyzer.load_model("weights.hdf5")

    if mode == 'predict':
        predict(sequence, input_len, analyzer, nb_predictions=nb_predictions,
                mapping=mapping, sentence_length=sentence_length)
        return mode

    # number of training sampes and validation samples
    nb_training_samples = batch_size * nb_batch
    nb_validation_samples = int(nb_training_samples * 0.05)

    # train model and output generated sequence
    for iteration in range(1, nb_iterations+1):
        # data generator of X_train and y_train, with random offset
        train_data = data_generator(sequence, input_len, mapping=mapping,
                                    sentence_length=sentence_length, step=step,
                                    offset=np.random.randint(0, step-1),
                                    batch_size=batch_size)

        # data generator of X_val and y _val, with random offset
        val_data = data_generator(sequence, input_len, mapping=mapping,
                                  sentence_length=sentence_length, step=step,
                                  offset=np.random.randint(0, step-1),
                                  batch_size=batch_size)
        print ""
        print "------------------------ Start Training ------------------------"
        print "Iteration: ", iteration

        # history of losses and accuracy
        history = History()

        # saves the model weights after each epoch
        # if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       verbose=1, save_best_only=True)

        # train the model with data generator
        analyzer.model.fit_generator(train_data,
                                     samples_per_epoch=nb_training_samples,
                                     nb_epoch=nb_epoch, verbose=1,
                                     show_accuracy=show_accuracy,
                                     callbacks=[history, checkpointer],
                                     validation_data=val_data,
                                     nb_val_samples=nb_validation_samples)

        # start index of the seed, random number in range
        start_index = np.random.randint(0, len(sequence) - sentence_length - 1)

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
                if mapping == 'o2o':
                    predictions = analyzer.model.predict(seed, verbose=0)[0]
                elif mapping == 'm2m':
                    predictions = analyzer.model.predict(seed,
                                                         verbose=0)[0][
                                                             sentence_length-1]
                # print "predictions length: %d" %len(predictions)
                next_id = sample(predictions, T)
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

    return mode


if __name__ == '__main__':
    train()
