"""
Simple Naive Bayes classifier implimentation for sequence prediction.

Author: Chang Liu (fluency03)
Data: 2016-05-12
"""


import glob
import numpy as np
from sklearn.naive_bayes import MultinomialNB


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
        self.build(window_size, nb_classes)

    def build(self, window_size, nb_classes):
        """
        Build up the matrix.
        """
        self.ny = np.zeros(nb_classes)
        self.nxy = np.zeros((window_size, nb_classes, nb_classes))
        self.py = np.zeros(self.nb_classes)

    def train(self, X, y):
        """
        Train the model.
        """
        for i in xrange(len(y)):
            self.ny[y[i]] += 1
            for j in xrange(self.window_size):
                self.nxy[j, X[i, j], y[i]] += 1

        N = np.sum(self.ny)

        # Prior
        for i in xrange(N):
            self.py[i] = ((self.ny[i] + self.alpha) /
                          (N + self.alpha * self.nb_classes))

    def evaluate(self, X, y):
        """
        Evaluate the model.
        """
        length = len(y)
        correct = 0
        pyx = np.zeros(self.nb_classes)

        for i in xrange(length):
            # Likelihood
            pxy = np.zeros(self.window_size)
            for j in xrange(self.window_size):
                pxy[j] = ((self.nxy[j, X[i, j], y[i]] + self.alpha) /
                          (self.ny[i] + self.alpha*self.nb_classes))
            # Posterior
            pyx[i] = self.py[i] * np.prod(pxy)
            # check the prediction
            if y[i] == np.argmax(pyx):
                correct += 1

        accuracy = (correct * 1.0) / length
        print "Accuracy: %.3f%%" %accuracy


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


def main(sentence_length=40):
    """
    Train the model.

    Arguments:
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
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

    X_train, y_train = get_data(train_sequence, sentence_length=sentence_length,
                                step=1, random_offset=False)
    X_val, y_val = get_data(val_sequence, sentence_length=sentence_length,
                            step=40, random_offset=False)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)
    print clf.predict(X_train[0:1])



if __name__ == '__main__':
    main()
