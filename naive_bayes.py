"""
This program analyze the integer sequence using Multinomial Naive Bayes based
on the python machine learning library sklearn.

Author: Chang Liu (fluency03)
Data: 2016-05-12
"""


import glob
import numpy as np
from sklearn.naive_bayes import MultinomialNB


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



def main(mapping='o2o', sentence_length=40, step=1):
    """
    Train the model.

    Arguments:
        sentence_length: {integer}, the length of each training sentence.
        step: {integer}, the sample steps.
        mapping: {string}, input to output mapping.
            'o2o': one-to-one
            'm2m': many-to-many
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

    X_train, y_train = get_data(train_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=False)
    X_val, y_val = get_data(val_sequence, input_len, mapping=mapping,
                            sentence_length=sentence_length, step=step,
                            random_offset=False)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)



if __name__ == '__main__':
    main()
