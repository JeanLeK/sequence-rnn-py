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
        self.nxy = np.zeros((self.window_size,
                             self.nb_classes,
                             self.nb_classes), dtype=np.int)
        self.py = np.zeros(self.nb_classes)

    def train(self, X, y):
        """
        Train the model.
        """
        N = len(y)
        for i in xrange(N):
            self.ny[y[i]] += 1
            for j in xrange(self.window_size):
                self.nxy[j, X[i, j], y[i]] += 1

        # print np.argmax(self.ny)

        # ------------------- Prior ------------------- #
        for i in xrange(N):
            self.py[y[i]] = ((self.ny[y[i]] + self.alpha) /
                             (N + self.alpha * self.nb_classes))

    def evaluate(self, X, y):
        """
        Evaluate the model.
        """
        length = len(y)
        print "length: %d " %length
        correct = 0

        for i in xrange(length):
            print "evaluating %d ..." %i
            # ------------------- Likelihood ------------------- #
            pxy = np.zeros((self.nb_classes, self.window_size))
            for p in xrange(self.nb_classes):
                for k in xrange(self.window_size):
                    pxy[p, k] = ((self.nxy[k, X[i, k], p] + self.alpha) /
                                 (self.ny[p] + self.alpha * self.nb_classes))
            # ------------------- Posterior ------------------- #
            pyx = np.zeros(self.nb_classes)
            for j in xrange(self.nb_classes):
                pyx[j] = self.py[j] * np.prod(pxy[j])

            # normalization
            # pyx_sum = np.sum(pyx)
            # pyx = np.asarray([pyx[p]/pyx_sum for p in xrange(self.nb_classes)])

            # check the prediction
            y_pred = np.argmax(pyx)
            print ("y_pred: %d , max_prod: %.3f%%, y_true_prob: %.3f%% ,"
                   %(y_pred, max(pyx)*100.0, pyx[y[i]]*100.0))
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
        step: {integer}, the sample steps.
    """
    # get parameters and dimensions of the model
    print "Loading training data..."
    train_sequence, input_len1 = get_sequence("./data")
    print "Loading validation data..."
    val_sequence, input_len2 = get_sequence("./data")
    input_len = max(input_len1, input_len2)

    print "Training sequence length: %d" %len(train_sequence)
    print "Validation sequence length: %d" %len(val_sequence)
    print "#classes: %d\n" %input_len

    X_train, y_train = get_data(train_sequence, sentence_length=sentence_length,
                                step=1, random_offset=False)
    X_val, y_val = get_data(val_sequence, sentence_length=sentence_length,
                            step=1, random_offset=False)

    start_time = time.time()

    nb = NaiveBayes(window_size=sentence_length,
                    nb_classes=input_len,
                    alpha=1.0/input_len)

    print "Train the model...\n"
    nb.train(X_train, y_train)

    print "Evaluate the model...\n"
    nb.evaluate(X_val, y_val)

    stop_time = time.time()
    print "Stop...\n"
    print "--- %s seconds ---\n" % (stop_time - start_time)

if __name__ == '__main__':
    main()
