"""
Simple Naive Bayes classifier implimentation for sequence prediction.

Author: Chang Liu (fluency03)
Data: 2016-05-12
"""

import cPickle as pickle
import glob
import os
import time
from math import log
import numpy as np
from rnn_sequence_analyzer import plot_hist, plot_and_write_prob


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

    def save_model(self, filename):
        """
        Save the model information to a file.
        """
        print "    |-Write the model into %s ..." %filename
        with open(filename, 'w') as pkl_file:
            pickle.dump({'ny': self.ny, 'nx_y': self.nx_y,
                         'window_size': self.window_size,
                         'nb_classes': self.nb_classes,
                         'alpha': self.alpha}, pkl_file)

    def load_model(self, filename):
        """
        Load the model information from a file.
        """
        if os.path.isfile(filename):
            print "%s existing, loading it...\n" %filename
            with open(filename) as pkl_file:
                model = pickle.load(pkl_file)
                self.ny = model['ny']
                self.nx_y = model['nx_y']
                # self.window_size = model['window_size']
                # self.nb_classes = model['nb_classes']
                # self.alpha = model['alpha']
        else:
            print "File does not exist!"

    def evaluate(self, X, y, normalization=True, log_scale=False):
        """
        Evaluate the model.

        Arguments:
            X: {array}, X evaluation data.
            y: {array}, y evaluation data.
            normalization: {bool}, whether do the normalization.
            log_scale: {bool}, whether transfer probabilities on log scale.
        """
        def scale(p):
            """
            Probability in log scale.
            """
            return log(p) if log_scale else p

        def normalize(py_x):
            """
            Normalize the probabilities.
            """
            py_x_sum = np.sum(py_x)
            return np.asarray([py_x[p] / py_x_sum
                               for p in xrange(self.nb_classes)])

        N = np.sum(self.ny)
        length = len(y)
        print "length: %d " %length
        correct = 0

        probs = np.zeros(length)
        if not log_scale:
            probs[:self.window_size] = 1.0

        # ------------------- Prior ------------------- #
        py = np.zeros(self.nb_classes)
        for i in xrange(self.nb_classes):
            py[i] = ((self.ny[i] + self.alpha) /
                     (N + self.alpha * self.nb_classes))

        for i in xrange(length):
            print "evaluating %d ..." %i
            # ------------------- Likelihood ------------------- #
            px_y = np.zeros((self.nb_classes, self.window_size))
            for p in xrange(self.nb_classes):
                for k in xrange(self.window_size):
                    px_y[p, k] = ((self.nx_y[k, X[i, k], p] +
                                   self.alpha) /
                                  (self.ny[p] +
                                   self.alpha * self.nb_classes))
            # ------------------- Posterior ------------------- #
            py_x = np.zeros(self.nb_classes)
            for j in xrange(self.nb_classes):
                py_x[j] = py[j] * np.prod(px_y[j])

            # ------------------- Normalization ------------------- #
            if normalization:
                py_x = normalize(py_x)

            # ------------------- Prediction ------------------- #
            # check the prediction
            y_pred = np.argmax(py_x)
            y_true = y[i]

            max_prob = scale(py_x[y_pred])
            print ("y_pred: %d , max_prod: %.8f, y_true_prob: %.8f ,"
                   %(y_pred, max_prob, scale(py_x[y_true])))

            if y_true == y_pred:
                correct += 1

            probs[i + self.window_size] = max_prob

        accuracy = (correct * 100.0) / length
        print "Accuracy: %.4f%%" %accuracy

        print "    |-Plot figures ..."
        plot_and_write_prob(probs,
                            "nb_prob_",
                            [0, 50000, 0, 1],
                            'Log' if log_scale else 'Normal')

    def evaluate_all(self, X, y, nb_options=3, normalization=True): # pylint: disable=R0912
        """
        Evaluate the model.

        Arguments:
            X: {array}, X evaluation data.
            y: {array}, y evaluation data.
            nb_options: {interger}, number of predicted options.
            normalization: {bool}, whether do the normalization.
        """
        N = np.sum(self.ny)
        length = len(y)
        print "length: %d " %length

        probs = np.zeros((nb_options+1, length + self.window_size))
        for o in xrange(nb_options+1):
            probs[o][:self.window_size] = 1.0

        # probability in negative log scale
        log_probs = np.zeros((nb_options+1, length + self.window_size))

        # count the number of correct predictions
        nb_correct = [0] * (nb_options+1)

        # ------------------- Prior ------------------- #
        py = np.zeros(self.nb_classes)
        for i in xrange(self.nb_classes):
            py[i] = ((self.ny[i] + self.alpha) /
                     (N + self.alpha * self.nb_classes))

        try:
            for i in xrange(length):
                print "evaluating %d ..." %i
                # ------------------- Likelihood ------------------- #
                px_y = np.zeros((self.nb_classes, self.window_size))
                for p in xrange(self.nb_classes):
                    for k in xrange(self.window_size):
                        px_y[p, k] = ((self.nx_y[k, X[i, k], p] +
                                       self.alpha) /
                                      (self.ny[p] +
                                       self.alpha * self.nb_classes))
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
                y_pred = np.argsort(py_x)[-nb_options:][::-1]
                y_true = y[i]
                print y_pred, y_true

                next_probs = [0.0] * (nb_options+1)
                next_probs[0] = py_x[y_true]

                for o in xrange(nb_options):
                    if y_true == y_pred[o]:
                        next_probs[o+1] = 1.0
                        nb_correct[o+1] += 1

                next_probs = np.maximum.accumulate(next_probs)
                print next_probs

                for k in xrange(nb_options+1):
                    probs[k, i + self.window_size] = next_probs[k]
                    # get the negative log probability
                    log_probs[k, i + self.window_size] = -log(next_probs[k])

        except:
            print "KeyboardInterrupt"

        nb_correct = np.add.accumulate(nb_correct)
        for n in xrange(nb_options+1):
            print "Accuracy %d: %.4f%%" %(n, (nb_correct[n] * 100.0 / (i + 1))) # pylint: disable=W0631

        print "    |-Plot figures ..."
        for q in xrange(nb_options+1):
            plot_and_write_prob(probs[q],
                                "nb_prob_"+str(q),
                                [0, 50000, 0, 1],
                                'Normal')
            plot_and_write_prob(log_probs[q],
                                "nb_log_prob_"+str(q),
                                [0, 50000, 0, 25],
                                'Log')
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
    max_value = 0

    for seqfile in seqfiles:
        sequence = []
        with open(seqfile, 'r') as f:
            one_sequence = [int(id_) for id_ in f]
            print "        %s, sequence length: %d" %(seqfile,
                                                      len(one_sequence))
            sequence.extend(one_sequence)
            total_length += len(one_sequence)
        max_new = np.amax(sequence)
        max_value = max_new if max_new > max_value else max_value
        sequences.append(sequence)

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = max_value + 2

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


def main(sentence_length=3, mode='train'):
    """
    Train the model.

    Arguments:
        sentence_length: {integer}, the length of each training sentence.
    """
    # get parameters and dimensions of the model
    print "Loading training data..."
    train_sequence, input_len1, total_length1 = get_sequence("./train_data/*")

    print "Loading validation data..."
    val_sequence, input_len2, total_length2 = get_sequence("./validation_data/*")

    input_len = max(input_len1, input_len2)

    print "Training sequence length: %d" %total_length1
    print "Validation sequence length: %d" %total_length2
    print "#classes: %d\n" %input_len

    start_time = time.time()

    nb = NaiveBayes(window_size=sentence_length,
                    nb_classes=input_len,
                    alpha=1.0/input_len)

    if mode == 'train':
        print "Train the model...\n"
        for sequence in train_sequence:
            X_train, y_train = get_data(sequence, sentence_length=sentence_length,
                                        step=1, random_offset=False)
            nb.train(X_train, y_train)
        # nb.save_model('2.pkl')
    elif mode == 'load':
        nb.load_model('2.pkl')

    print "Evaluate the model...\n"
    # for sequence in val_sequence:
    #     X_val, y_val = get_data(sequence, sentence_length=sentence_length,
    #                             step=1, random_offset=False)
    #     nb.evaluate(X_val, y_val, normalization=True, log_scale=False)

    for sequence in val_sequence:
        X_val, y_val = get_data(sequence, sentence_length=sentence_length,
                                step=1, random_offset=False)
        nb.evaluate_all(X_val, y_val, nb_options=3, normalization=True)

    stop_time = time.time()
    print "Stop...\n"
    print "--- %s seconds ---\n" % (stop_time - start_time)

if __name__ == '__main__':
    main()
