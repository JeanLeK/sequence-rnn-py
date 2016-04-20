"""
Markov Chain, a comparable model to RNN as a baseline.

The LabeledMarkovPredictor is riginally written by Erik Ylipaa at SICS.

Author: Chang Liu (fluency03)
Data: 2016-04-15
"""

# import unittest
# from collections import Counter, defaultdict

import glob
import numpy as np
from hmmlearn.hmm import MultinomialHMM


class LabeledMarkovPredictor(object):
    """
    Model which builds a first order markov model of the labeled data.
    """
    def __init__(self, num_classes, # pylint: disable=W0613
                 eval_during_training=False, **kwargs):
        """
        Create a new LabeledMarkov predictor.

        Arguments:
            num_classes: {integer}, the number of class labels in the data.
            eval_during_training: {bool}, if True, loss will be calculated
                during training. For the Markov chain this means very little.
                Disabling this speeds up training.
            kwargs:
        """
        self.num_classes = num_classes
        self.eval_during_training = eval_during_training
        self.dirty_counts = True  #
        self.setup_params()

    def setup_params(self):
        """
        Set up other perematers. The model is basically just a matrix. Each row
        of the matrix is the conditional probabilty for the next symbol in the
        sequence, given the current symbol. We set all entries to 1, giving us
        a uniform distribution as a prior.
        """
        # the matrix initialized with all ones
        self.W = np.ones((self.num_classes, self.num_classes), np.uint64)

        # Give the class count two dimensions, but put the second to 1, so it's
        # broadcastable over W when we wish to divide. Set to to the number of
        # classes, so that it will give us the uniform distribution as a prior
        self.class_counts = np.full(self.num_classes,
                                    self.num_classes,
                                    dtype=np.uint64)

        self.log_class_counts = np.log(np.full(self.num_classes,
                                               self.num_classes,
                                               dtype=np.uint64))
        self.dirty_counts = False

    def train(self, training_arguments, *args, **kwargs): # pylint: disable=W0613
        """
        Updates the model based on the input batch. The input should be a tuple
        of two ndarray training-batches.

        Arguments:
            training_arguments: {tuple}, should be a tuple of x- and y-batches.
                (x_batch, y_bath). The batches should be ndarray matrices of
                integer labels. The first dimension is the time dimension, the
                second the batch dimension. The shape is considered to have the
                semantics: (sequence_length, batch_size).
            args:
            kwargs:
        Returns: {tuple}, (training_loss, info_dict). The training loss will
            be the average negative log of the probability of the y_batch before
            training on the x_batch. The info_dict is an empty dictionary for
            this model. If eval_during_training was set to False when the model
            was instantiated, None is returned instead of the loss.
        """
        # We disregard any arguments except the training arguments tuple
        try:
            x_batch, y_batch, mask = training_arguments # pylint: disable=W0612
        except ValueError:
            x_batch, y_batch = training_arguments
            mask = None

        sequence_length, batch_size = x_batch.shape

        # We go over each timestep and increase all the columns denoted by the
        # y's for the rows denoted by the x's
        for t in range(sequence_length):
            for batch_num in range(batch_size):
                x = x_batch[t, batch_num]
                y = y_batch[t, batch_num]
                self.W[x, y] += 1
                self.class_counts[x] += 1
                self.dirty_counts = True

        info_dict = dict()
        if self.eval_during_training:
            loss = self.evaluate(training_arguments)
        else:
            loss = None
        return loss, info_dict

    def evaluate(self, training_argument):
        """
        Get the average negative log probability for the y_batch, using the
        model predicted probabilities from the x_batch.

        Arguments:
            training_argument: {tuple}, a pair of ndarrays (x_batch, y_batch).
                The batches should be matrices of integers of the same shape,
                where the first dimension is time, the second is over batches.
        Returns: {float}, The average negative log probability the model
            assigned the correct answers of the y_batch given the x_batch.
        """
        # We disregard any arguments except the training arguments tuple
        try:
            x_batch, y_batch, mask = training_argument # pylint: disable=W0612
        except ValueError:
            x_batch, y_batch = training_argument
            mask = None

        x_batch = x_batch.astype(np.int)
        y_batch = y_batch.astype(np.int)
        sequence_length, batch_size = x_batch.shape

        # np.seterr(divide='ignore'). We ignore division by zero, since we will
        # be performing many of them. We will return the negative log likelihood
        # per sequence. This will be the logarithm of normalized value for each
        # of the entries in the matrix. The matrix needs to be normalized by row
        # P = np.divide(self.W, self.class_counts)
        # P[np.where(np.isnan(P))] = 1/self.num_classes
        # Any rows with NaN, we replace with a uniform score
        flat_x = x_batch.flatten()
        flat_y = y_batch.flatten()
        if self.dirty_counts:
            self.log_class_counts = np.log(self.class_counts)
            self.dirty_counts = False

        # We should take the negative log of the probabilities, this is the same
        # as taking the log of the W[x,y]/count[x], which is the same as
        # log(W[x,y]) - log(count[x])
        # probs = self.W[flat_x, flat_y] / self.class_counts[flat_x]
        # log_probs = np.log(probs)
        log_probs = (np.log(self.W[flat_x, flat_y]) -
                     self.log_class_counts[flat_x])
        loss = - float(np.sum(log_probs))
        # for consistency, divide the negative log loss with the batch size and
        # sequence length returning the same loss as the RNN models
        sequence_loss = loss / (batch_size * sequence_length)
        return sequence_loss

    def predict(self, x_batch):
        """
        Arguments:
            x_batch: {np.array}, An ndarray of integer labels.
        Returns: {integer}. The predicted label the same shape as x_batch.
        """
        x_batch = x_batch.astype(np.int)
        # for each entry in x_batch, it will pick out a row for W.
        label_counts = self.W[x_batch]
        # along each row picked by the x_batch
        # return the index of the highest count
        return np.argmax(label_counts, axis=-1)


def transpose(theList):
    """
    Transpose matrix for Markov Chain model.

    Arguments:
        theList: {list}, the input list.
    Returns: {np.array}, the transposed np.array.
    """
    return np.asarray(theList).transpose()


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


def get_data(sequence, sentence_length=40, random_offset=False):
    """
    Retrieves data from a plain txt file and formats it using one-hot vector.

    Arguments:
        sequence: {lsit}, the original input sequence
        sentence_length: {integer}, the length of each training sentence.
        random_offset: {bool}, the offset is random between step or is 0.
    Returns:
        {list}, training input data X
        {list}, training target data y
    """
    X_sentences = []
    y_sentences = []

    offset = np.random.randint(0, sentence_length) if random_offset else 0

    # creat batch data and next sentences
    for i in range(offset, len(sequence) - sentence_length, sentence_length):
        X_sentences.append(sequence[i : i + sentence_length])
        y_sentences.append(sequence[i + 1 : i + sentence_length + 1])

    return X_sentences, y_sentences


def train(sentence_length=40):
    """
    Train the markov chain.

    Arguments:
        sentence_length: {integer}, length of one sentence in the data set.
    """
    # get parameters and dimensions of the model
    print "Loading training data..."
    train_sequence, input_len1 = get_sequence("./train_data/*")
    print "Loading validation data..."
    val_sequence, input_len2 = get_sequence("./validation_data/*")
    nb_classes = max(input_len1, input_len2)

    print "Training sequence length: %d" %len(train_sequence)
    print "Validation sequence length: %d" %len(val_sequence)
    print "#classes: %d\n" %nb_classes

    X_train, y_train = get_data(train_sequence,
                                sentence_length=sentence_length,
                                random_offset=False)
    X_val, y_val = get_data(val_sequence,
                            sentence_length=sentence_length,
                            random_offset=False)

    print "Build Markov Chain..."
    model = LabeledMarkovPredictor(nb_classes)

    print "Train the model..."
    model.train((transpose(X_train), transpose(y_train)))

    print "Validating..."
    validation_loss = 0
    validation_loss = model.evaluate((transpose(X_val), transpose(y_val)))

    print "Validation loss: {}".format(validation_loss)


# TODO: not working yet
def train_hmm():
    """
    HMM for sequence learning.
    """
    print "Loading training data..."
    train_sequence, num_classes = get_sequence("./train_data/*")

    print "Build HMM..."
    model = MultinomialHMM(n_components=2)

    print "Train HMM..."
    model.fit([train_sequence])



if __name__ == '__main__':
    train()
    # train_hmm()
