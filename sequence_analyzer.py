"""
This program analyze the integer sequence using Recurrent Neural Network (RNN)
with Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) based on the
python library Keras.

"Keras is a minimalist, highly modular neural networks library, written in
 Python and capable of running on top of either TensorFlow or Theano."

                                                ---- Keras (http://keras.io/)


It is based on this Keras example - lstm_text_generation:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py


Author: Chang Liu (fluency03)
Data: 2016-03-17
"""


from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
# from keras.utils.visualize_util import plot
import numpy as np
import random
import sys


class SequenceAnalyzer(object):
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

    @classmethod
    def sample(cls, prob, temperature=1.0):
        """
        softmax function for reinforcement learning
        """
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.argmax(np.random.multinomial(1, prob, 1))



class LossHistory(Callback):
    """
    Record the loss and accuracy history
    """
    def on_train_begin(self, logs={}):
        # self.losses = []
        # training loss and accuracy
        self.train_losses = []
        self.train_acc = []
        # validation loss and accuracy
        self.val_losses = []
        self.val_acc = []

    # def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        # record training loss and accuracy
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        # record validation loss and accuracy
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))



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



def print_losses(history):
    # print the losses and accuracy of training
    print "losses and accuracy of Training: "
    train_losses = history.train_losses
    train_acc = history.train_acc
    for l, a in zip(train_losses, train_acc):
        print "     Loss: %.4f, Accuracy: %.4f" %(l, a)

    # print the losses and accuracy of validation
    print "losses and accuracy of Validation: "
    val_losses = history.val_losses
    val_acc = history.val_acc
    for l, a in zip(val_losses, val_acc):
        print "     Loss: %.4f, Accuracy: %.4f" %(l, a)


def train():
    """
    Trains the network and outputs the generated text.
    Trains using batch size of 100, 60 epochs total.
    """
    # get parameters and dimensions of the model
    sequence, sentence_length, input_len, x, y = get_data()

    # the size of each hidden layer
    hidden_len = 512

    # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
    rnn = SequenceAnalyzer(sentence_length, input_len, hidden_len, input_len)

    print "Building Model..."
    rnn.build_lstm(dropout=0.2)

    # save the model weight into a file
    # rnn.model.save_weights('my_model_weights.h5')

    # plot the model, need the following packages:
    # pydot, graphviz, setuptools, pyparsing
    # plot(rnn.model, to_file='rnn_model.png')

    nb_iterations = 40
    # train model and output generated sequence
    for iteration in range(1, nb_iterations+1):
        print ""
        print "------------------------ Start Training ------------------------"
        print "Iteration: ", iteration

        # history of losses and accuracy
        history = LossHistory()

        # saves the model weights after each epoch
        # if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       verbose=1, save_best_only=True)

        # train the model
        rnn.model.fit(x, y, batch_size=128, nb_epoch=1, validation_split=0.1,
                      show_accuracy=True, verbose=1,
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
                predictions = rnn.model.predict(seed, verbose=1)[0]
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
            print ""

        # print the losses and accuracy
        print_losses(history)


if __name__ == '__main__':
    train()
