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

import glob
# import os
import sys
import csv
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Input, Activation, Dense, Dropout, LSTM, GRU, merge
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model
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
        print "Save Weights %s ..." %filename
        self.model.save_weights(filename)

    def load_model(self, filename):
        """
        Load the model weight into a hdf5 file.

        Arguments:
            filename: {string}, the name/path to the file
                to which the weights are going to be loaded.
        """
        print "Load Weights %s ..." %filename
        self.model.load_weights(filename)

    def plot_model(self, filename):
        """
        Plot model.

        Arguments:
            filename: {string}, the name/path to the file
                to which the model graphic is plotted.
        """
        print "Plot Model %s ..." %filename
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
    def build(self, layer='LSTM', mapping='m2m', nb_layers=2, dropout=0.2):
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
        print ("    layer = %d-%s , mapping = %s , "
               "nb_layers = %d , dropout = %.2f"
               %(self.hidden_len, layer, mapping, nb_layers, dropout))

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

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])


class BRNN(SequenceAnalyzer):
    """
    Bi-directional RNN model of the sequence analyzer. Graph Model.
    """
    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        super(BRNN, self).__init__(sentence_length,
                                   input_len, hidden_len, output_len)

    @override
    def build(self, layer='LSTM', mapping='m2m', nb_layers=2, dropout=0.2):
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
        print ("    layer = %d-%s , mapping = %s , "
               "nb_layers = %d , dropout = %.2f"
               %(self.hidden_len, layer, mapping, nb_layers, dropout))

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
        input_layer = Input(shape=(self.sentence_length, self.input_len),
                            dtype='float32')

        # first Bi-directional LSTM layer
        forward1 = LAYER(self.hidden_len,
                         return_sequences=return_sequences[0])(input_layer)
        forward_dropout1 = Dropout(dropout)(forward1) # pylint: disable=W0612
        backward1 = LAYER(self.hidden_len,
                          return_sequences=return_sequences[0],
                          go_backwards=True)(input_layer)
        backward_dropout1 = Dropout(dropout)(backward1) # pylint: disable=W0612

        # following Bi-directional layers
        for nl in range(nb_layers-1):
            exec("%s = LAYER(self.hidden_len, return_sequences=%s)(%s)"
                 %('forward' + str(nl+2),
                   return_sequences[nl+1],
                   'forward_dropout' + str(nl+1)))
            exec("%s = Dropout(dropout)(%s)"
                 %('forward_dropout' + str(nl+2),
                   'forward' + str(nl+2)))
            exec(("%s = LAYER(self.hidden_len, return_sequences=%s, "
                  "go_backwards=True)(%s)")
                 %('backward' + str(nl+2),
                   return_sequences[nl+1],
                   'backward_dropout' + str(nl+1)))
            exec("%s = Dropout(dropout)(%s)"
                 %('backward_dropout' + str(nl+2),
                   'backward' + str(nl+2)))

        merged_layer = merge([locals()['forward_dropout' + str(nb_layers)],
                              locals()['backward_dropout' + str(nb_layers)]],
                             mode='concat', concat_axis=-1)

        if mapping == 'o2o':
            output_layer = Dense(self.output_len,
                                 activation='softmax')(merged_layer)
        elif mapping == 'm2m':
            output_layer = TimeDistributed(
                Dense(self.output_len, activation='softmax'))(merged_layer)

        # add ouput
        self.model = Model(input=input_layer, output=output_layer)

        # try using different optimizers and different optimizer configs
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])


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


def data_generator(sequence, vocab_size, mapping='m2m', sentence_length=40,
                   step=3, random_offset=True, batch_size=128):
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
        random_offset: {bool}, the offset is random between step or is 0.
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
        offset = np.random.randint(0, step) if random_offset else 0
        for i in range(offset, len(sequence) - sentence_length, step):
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


def train(analyzer, train_data, nb_training_samples,
          val_data, nb_validation_samples,
          nb_epoch=50, nb_iterations=4):
    """
    Trains the network.

    Arguments:
        analyzer: {SequenceAnalyzer}.
        train_data: {tuple}, training data (X_train, y_train).
        val_data: {tuple}, validation data (X_val, y_val).
        nb_training_samples: {integer}, the number training samples.
        nb_validation_samples: {integer}, the number validation samples.
        nb_iterations: {integer}, number of iterations.
        sentence_length: {integer}, the length of each training sentence.
    """
    for iteration in range(1, nb_iterations+1):
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

        # train the model with data generator
        analyzer.model.fit_generator(train_data,
                                     samples_per_epoch=nb_training_samples,
                                     nb_epoch=nb_epoch, verbose=1,
                                     callbacks=[history, checkpointer],
                                     validation_data=val_data,
                                     nb_val_samples=nb_validation_samples)

        analyzer.save_model("weights-after-iteration.hdf5")


def run(hidden_len=512, batch_size=128, nb_batch=200, nb_epoch=50,
        nb_iterations=4, lr=0.001, validation_split=0.05, nb_predictions=20,
        mapping='m2m', sentence_length=80, step=80, mode='train'):
    """
    Train, evaluate, or predict.

    Arguments:
        hidden_len: {integer}, the size of a hidden layer.
        batch_size: {interger}, the number of sentences per batch.
        nb_batch: {integer}, number of batches to be trained durign each epoch.
        nb_epoch: {interger}, number of epoches per iteration.
        nb_iterations: {integer}, number of iterations.
        lr: {float}, learning rate.
        validation_split: {float} (0 ~ 1), percentage of validation data
            among training data.
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

    # data generator of X_train and y_train, with random offset
    train_data = data_generator(train_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=True, batch_size=batch_size)

    # data generator of X_val and y _val,  with random offset
    val_data = data_generator(val_sequence, input_len, mapping=mapping,
                              sentence_length=sentence_length, step=step,
                              random_offset=True, batch_size=batch_size)

    # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
    analyzer = SequenceAnalyzer(sentence_length,
                                input_len, hidden_len, input_len)

    # build model
    analyzer.build(layer='LSTM', mapping=mapping, nb_layers=2, dropout=0.2)

    # plot model
    # analyzer.plot_model()

    # load the previous model weights
    # analyzer.load_model("weightsf4-61.hdf5")

    # reset the learning rate
    if lr != 0.001:
        analyzer.model.optimizer.lr.set_value(lr)

    if mode == 'predict':
        predict(val_sequence, input_len, analyzer, nb_predictions=nb_predictions,
                mapping=mapping, sentence_length=sentence_length)
    elif mode == 'evaluate':
        print "Metrics: " + ', '.join(analyzer.model.metrics_names)
        X_val, y_val = data_generator(val_sequence, input_len, mapping=mapping,
                                      sentence_length=sentence_length,
                                      step=step, random_offset=False,
                                      batch_size=batch_size)
        results = analyzer.model.evaluate(X_val, y_val, #pylint: disable=W0612
                                          batch_size=batch_size,
                                          verbose=1)
        print "Loss: ", results[0]
        print "Accuracy: ", results[1]
    elif mode == 'train':
        # number of training sampes and validation samples
        nb_training_samples = batch_size * nb_batch
        nb_validation_samples = int(nb_training_samples * validation_split)

        try:
            train(analyzer, train_data, nb_training_samples,
                  val_data, nb_validation_samples,
                  nb_epoch=nb_epoch, nb_iterations=nb_iterations)
        except KeyboardInterrupt:
            analyzer.save_model("weights-stop.hdf5")
    else:
        print "The mode = %s is not correct!!!" %mode

    return mode


if __name__ == '__main__':
    run()
