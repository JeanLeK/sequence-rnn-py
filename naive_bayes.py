"""
Simple Naive Bayes classifier implimentation for sequence prediction.

Author: Chang Liu (fluency03)
Data: 2016-05-12
"""


import glob
import time
from math import log
import numpy as np


class NaiveBayes(object):
    """
    Simple Naive Bayes classifier implimentation for sequence prediction.
    """
    def __init__(self, window_size, nb_classes, alpha=1.0):
        """
        Initialization. Set up some parameters. Build up the matrix.

        Arguments:
            window_size: {integer}, the size of input window.
            nb_classes: {integer}, number of uniques classes.
            alpha: {float}, the smoothing priors alpha >= 0 accounts for
                features not present in the learning samples and prevents zero
                probabilities in further computations. Setting alpha = 1 is
                called Laplace smoothing, while alpha < 1 is called
                Lidstone smoothing.

        """
        self.window_size = window_size
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.build()

    def build(self):
        """
        Build up the matrix.
        """
        self.ny = np.zeros((self.nb_classes,), dtype=np.int)
        self.nx_y = np.zeros((self.window_size,
                              self.nb_classes,
                              self.nb_classes), dtype=np.int)

    def train(self, X, y):
        """
        Train the model.

        Arguments:
            X: {array}, X training data.
            y: {array}, y training data.
        """
        N = len(y)
        for i in xrange(N):
            self.ny[y[i]] += 1
            for j in xrange(self.window_size):
                self.nx_y[j, X[i, j], y[i]] += 1

    def evaluate(self, X, y, normalization=False, log_scale=False):
        """
        Evaluate the model.

        Arguments:
            X: {array}, X evaluation data.
            y: {array}, y evaluation data.
            normalization: {bool}, whether do the normalization.
            log_scale: {bool}, whether transfer probabilities on log scale.
        """
        N = np.sum(self.ny)
        length = len(y)
        print "length: %d " %length
        correct = 0

        # ------------------- Prior ------------------- #
        py = np.zeros(self.nb_classes)
        for i in xrange(N):
            py[y[i]] = ((self.ny[y[i]] + self.alpha) /
                        (N + self.alpha * self.nb_classes))

        for i in xrange(length):
            print "evaluating %d ..." %i
            # ------------------- Likelihood ------------------- #
            px_y = np.zeros((self.nb_classes, self.window_size))
            for p in xrange(self.nb_classes):
                for k in xrange(self.window_size):
                    px_y[p, k] = ((self.nx_y[k, X[i, k], p] + self.alpha) /
                                  (self.ny[p] + self.alpha * self.nb_classes))
            # ------------------- Posterior ------------------- #
            py_x = np.zeros(self.nb_classes)
            for j in xrange(self.nb_classes):
                py_x[j] = py[j] * np.prod(px_y[j])

            # ------------------- Normalization ------------------- #
            if normalization:
                py_x_sum = np.sum(py_x)
                py_x = np.asarray([py_x[p] / py_x_sum
                                   for p in xrange(self.nb_classes)])

            # ------------------- Prediction ------------------- #
            # check the prediction
            y_pred = np.argmax(py_x)
            print ("y_pred: %d , max_prod: %.3f%%, y_true_prob: %.3f%% ,"
                   %(y_pred, max(py_x)*100.0, py_x[y[i]]*100.0))
            if y[i] == y_pred:
                correct += 1

        accuracy = (correct * 100.0) / length
        print "Accuracy: %.3f%%" %accuracy

        return accuracy

    def predict(self, X):
        """
        Predict next sequence.
        """
        pass



def get_sequence(filepath):
    """
    Get the original sequence from file.

    Arguments:
        filename: {string}, the name/path of input log sequence file.
    Returns:
        {list}, the log sequence.
        {integer}, the size of vocabulary.
        {integer}, total length of the sequences.
    """
    # read file and convert ids of each line into array of numbers
    seqfiles = glob.glob(filepath)
    sequences = []
    total_length = 0

    for seqfile in seqfiles:
        sequence = []
        with open(seqfile, 'r') as f:
            one_sequence = [int(id_) for id_ in f]
            print "        %s, sequence length: %d" %(seqfile,
                                                      len(one_sequence))
            sequence.extend(one_sequence)
            total_length += len(one_sequence)
        sequences.append(sequence)

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = np.amax(sequences) + 2

    return sequences, vocab_size, total_length


def get_data(sequence, sentence_length=40, step=3, random_offset=True):
    """
    Retrieves data from a plain txt file and formats it using one-hot vector.

    Arguments:
        sequence: {lsit}, the original input sequence
        vocab_size: {integer}, the number of unique id classes
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        random_offset: {bool}, the offset is random between step or is 0.
    Returns:
        {np.array}, training input data X
        {np.array}, training target data y
    """
    X_sentences = []
    next_ids = []

    offset = np.random.randint(0, step) if random_offset else 0

    # creat batch data and next sentences
    for i in range(offset, len(sequence) - sentence_length, step):
        X_sentences.append(sequence[i : i + sentence_length])
        next_ids.append(sequence[i + sentence_length])

    # number of sampes
    # nb_samples = len(X_sentences)
    # print "total # of sentences: %d" %nb_samples

    return np.asarray(X_sentences), np.asarray(next_ids)


def main(sentence_length=3):
    """
    Train the model.

    Arguments:
        sentence_length: {integer}, the length of each training sentence.
    """
    # get parameters and dimensions of the model
    print "Loading training data..."
    train_sequence, input_len1, total_length1 = get_sequence("./train_data/*")

    print "Loading validation data..."
    val_sequence, input_len2, total_length2 = get_sequence("./train_data/*")

    input_len = max(input_len1, input_len2)

    print "Training sequence length: %d" %total_length1
    print "Validation sequence length: %d" %total_length2
    print "#classes: %d\n" %input_len

    start_time = time.time()

    nb = NaiveBayes(window_size=sentence_length,
                    nb_classes=input_len,
                    alpha=1.0/input_len)

    print "Train the model...\n"
    for sequence in train_sequence:
        X_train, y_train = get_data(sequence, sentence_length=sentence_length,
                                    step=1, random_offset=False)
        nb.train(X_train, y_train)

    print "Evaluate the model...\n"
    for sequence in val_sequence:
        X_val, y_val = get_data(sequence, sentence_length=sentence_length,
                                step=1, random_offset=False)
        nb.evaluate(X_val, y_val, normalization=False)

    stop_time = time.time()
    print "Stop...\n"
    print "--- %s seconds ---\n" % (stop_time - start_time)

if __name__ == '__main__':
    main()
