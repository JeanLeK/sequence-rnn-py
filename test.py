

# with open( "/home/cliu/Documents/SC-1/sequence", 'r') as f:
#     sequence = [int(x) for x in f]
#
# length = 20
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(sequence) - length, step):
#     sentences.append(sequence[i: i + length])
#     next_chars.append(sequence[i + length])
# print("total # of sentences: ", len(sentences))


# for i in range(0, 10, 3):
#     print i


# alist = [1, 2, 3, 4, 5]
#
# alist.pop(0)
# alist.append(6)
#
# print alist

# with open( "sequence", 'w') as f:
#     for i in range(10):
#         for j in range(1, 1656):
#             f.write(str(j) + '\n')
#
# list1 = [1, 2, 3]
# list2 = [0, 0, 0]
#
# print list2 + list1
#
# class Parent(object):
#     def __init__(self):
#         self.a = 1
#         self.b = 2
#     def printa(self):
#         print self.a

# class Child(Parent):
#     def __init__(self):
#         super(Child, self).__init__()
#     def printa(self):
#         print self.a
#
# child = Child()
#
# child.printa()

# print child.a


# a = [1, 2]
# b = [3, 4]
# c = [5, 6]
# d = [7, 8]
#
# total = []
#
# rows = zip(a, b, c, d)
#
# for i in range(len(a)):
#     total.append([a[i], b[i], c[i], d[i]])
#
# # print total[0].next()
# import csv
#
# with open('t.csv', 'wb') as csvfile:
#     his_writer = csv.writer(csvfile)
#     for row in rows:
#         his_writer.writerow(row)
#
# def get_data_o2o(sentence_length=40, step=3):
#     """
#     Retrieves data from a plain txt file and formats it using one-hot vector.
#     """
#     # read file and convert ids of each line into array of numbers
#     with open("/home/cliu/Documents/SC-1/sequence", 'r') as f:
#         sequence = [int(id_) for id_ in f]
#
#     # add two extra positions for 'unknown-log' and 'no-log'
#     vocab_size = max(sequence) + 2
#
#     # list of sentences
#     sentences = []
#     # list of the next id for each of the according sentence
#     next_ids = []
#
#     # creat batch data and next id sequences
#     for i in range(0, len(sequence) - sentence_length, step):
#         sentences.append(sequence[i: i + sentence_length])
#         next_ids.append(sequence[i + sentence_length])
#
#     # number of sampes
#     nb_samples = len(sentences)
#     print "total # of sentences: %d" %nb_samples
#
#     # one-hot vector (all zeros except for a single one at
#     # the exact postion of this id number)
#     X_train = np.zeros((nb_samples, sentence_length, vocab_size), dtype=np.bool)
#     # expected outputs for each sentence
#     y_train = np.zeros((nb_samples, vocab_size), dtype=np.bool)
#
#     for i, sentence in enumerate(sentences):
#         for t, id_ in enumerate(sentence):
#             # mark the each corresponding character in a sentence as 1
#             X_train[i, t, id_] = 1
#         # mark the corresponding character in expected output as 1
#         y_train[i, next_ids[i]] = 1
#
#     return sequence, sentence_length, vocab_size, X_train, y_train

# import random
#
# data = [1, 2, 3, 4, 5, 6, 7, 8]
#
# length = len(data)
#
# for i in range(length):
#     index = random.randint(0, length-i-1)
#     print data[index]
#     data = data[:index] + data[index+1:]

# def seq_generator(mapping='o2o', sentence_length=40, step=3, offset=0):
#     with open("test_data", 'r') as f:
#         sequence = [int(id_) for id_ in f]
#
#     # add two extra positions for 'unknown-log' and 'no-log'
#     vocab_size = max(sequence) + 2
#
#     # creat batch data and next sentences
#     for i in range(0, len(sequence) - sentence_length, step):
#         X_sentence = sequence[i : i + sentence_length]
#         if mapping == 'o2o':
#             # if mapping is one-to-one
#             next_id = sequence[i + sentence_length]
#         elif mapping == 'm2m':
#             # if mapping is many-to-many
#             y_sentence = sequence[i + 1 : i + sentence_length + 1]
#
#         # one-hot vector (all zeros except for a single one at
#         # the exact postion of this id number)
#         X_train = np.zeros((sentence_length, vocab_size), dtype=np.bool)
#         # expected outputs for each sentence
#         if mapping == 'o2o':
#             # if mapping is one-to-one
#             y_train = np.zeros((vocab_size), dtype=np.bool)
#         elif mapping == 'm2m':
#             # if mapping is many-to-many
#             y_train = np.zeros((sentence_length, vocab_size), dtype=np.bool)
#
#         for t, id_ in enumerate(X_sentence):
#             # mark the each corresponding character in a sentence as 1
#             X_train[t, id_] = 1
#             # if mapping is many-to-many
#             if mapping == 'm2m':
#                 y_train[t, y_sentence[t]] = 1
#         # if mapping is one-to-one
#         # mark the corresponding character in expected output as 1
#         if mapping == 'o2o':
#             y_train[next_id] = 1
#
#         yield X_train, y_train


#
# g = seq_generator()
#
# print g.next()

# for _ in range(5):
#     print g.next()

# class LAYER():
#     def __init__(self, a, b=0):
#         self.a = a
#         self.b = b
#
#
# class L(LAYER):pass
#
# l = L(1)
#
# print l.a, l.b
#
#
# def build_gru(self, mapping='o2o', nb_layers=2, dropout=0.2):
#     """
#     Bidirectional GRU with specified dropout rate (default 0.2), built with
#     softmax activation, cross entropy loss and rmsprop optimizer.
#     """
#     print "Building Model..."
#     # check whether the last layer return sequences
#     if mapping == 'o2o':
#         # if mapping is one-to-one
#         return_sequences = False
#     elif mapping == 'm2m':
#         # if mapping is many-to-many
#         return_sequences = True
#
#     self.model.add_input(input_shape=(self.sentence_length, self.input_len),
#                          name='input', dtype='float')
#
#     # first Bi-directional LSTM layer
#     self.model.add_node(GRU(self.hidden_len, return_sequences=True),
#                         name='forward1', input='input')
#     self.model.add_node(Dropout(dropout),
#                         name='forward_dropout1', input='forward1')
#     self.model.add_node(GRU(self.hidden_len, return_sequences=True,
#                             go_backwards=True),
#                         name='backward1', input='input')
#     self.model.add_node(Dropout(dropout),
#                         name='backward_dropout1', input='backward1')
#
#     # following Bi-directional GRU layers
#     for nl in range(nb_layers-1):
#         # check whether return sequences
#         if nl != nb_layers-2:
#             return_sequences_ = True
#         else:
#             return_sequences_ = return_sequences
#         # build following hidden layers
#         self.model.add_node(GRU(self.hidden_len,
#                                 return_sequences=return_sequences_),
#                             name='forward' + str(nl+2),
#                             input='forward_dropout' + str(nl+1))
#         self.model.add_node(Dropout(dropout),
#                             name='forward_dropout' + str(nl+2),
#                             input='forward' + str(nl+2))
#         self.model.add_node(GRU(self.hidden_len,
#                                 return_sequences=return_sequences_,
#                                 go_backwards=True),
#                             name='backward' + str(nl+2),
#                             input='backward_dropout' + str(nl+1))
#         self.model.add_node(Dropout(dropout),
#                             name='backward_dropout' + str(nl+2),
#                             input='backward' + str(nl+2))
#
#     # self.model.add_node(Dropout(dropout), name='dropout',
#                         # inputs=['forward', 'backward'])
#     self.model.add_node(Dense(self.output_len, activation='softmax'),
#                         name='softmax',
#                         inputs=['forward_dropout' + str(nb_layers),
#                                 'backward_dropout' + str(nb_layers)])
#     self.model.add_output(name='output', input='softmax')
#
#     # try using different optimizers and different optimizer configs
#     self.model.compile(loss={'output': 'categorical_crossentropy'},
#                        optimizer='rmsprop')

# def build_gru(self, mapping='o2o', nb_layers=2, dropout=0.2):
#     """
#     Stacked GRU with specified dropout rate (default 0.2), built with
#     softmax activation, cross entropy loss and rmsprop optimizer.
#
#     Arguments:
#         mapping: {string}, input to output mapping
#             'o2o': one-to-one
#             'm2m': many-to-many
#         nb_layers: {integer}, number of layers in total
#         dropout: {float}, dropout value
#     """
#     print "Building Model..."
#
#     # check whether the last layer return sequences
#     if mapping == 'o2o':
#         # if mapping is one-to-one
#         return_sequences = False
#     elif mapping == 'm2m':
#         # if mapping is many-to-many
#         return_sequences = True
#
#     # 2 layer GRU with specified number of nodes in the hidden layer.
#     self.model.add(GRU(self.hidden_len, return_sequences=True,
#                        input_shape=(self.sentence_length,
#                                     self.input_len)))
#     self.model.add(Dropout(dropout))
#
#     for nl in range(nb_layers-1):
#         # check whether return sequences
#         if nl != nb_layers-2:
#             return_sequences_ = True
#         else:
#             return_sequences_ = return_sequences
#         # build hidden layers
#         self.model.add(GRU(self.hidden_len,
#                            return_sequences=return_sequences_))
#         self.model.add(Dropout(dropout))
#
#     if mapping == 'o2o':
#         # if mapping is one-to-one
#         self.model.add(Dense(self.output_len))
#     elif mapping == 'm2m':
#         # if mapping is many-to-many
#         self.model.add(TimeDistributedDense(self.output_len))
#
#     self.model.add(Activation('softmax'))
#
#     self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# def build_gru(self, mapping='o2o', nb_layers=2, dropout=0.2):
#     """
#     Stacked GRU with specified dropout rate (default 0.2), built with
#     softmax activation, cross entropy loss and rmsprop optimizer.
#
#     Arguments:
#         mapping: {string}, input to output mapping
#             'o2o': one-to-one
#             'm2m': many-to-many
#         nb_layers: {integer}, number of layers in total
#         dropout: {float}, dropout value
#     """
#     print "Building Model..."
#
#     # check whether the last layer return sequences
#     if mapping == 'o2o':
#         # if mapping is one-to-one
#         return_sequences = False
#     elif mapping == 'm2m':
#         # if mapping is many-to-many
#         return_sequences = True
#
#     # 2 layer GRU with specified number of nodes in the hidden layer.
#     self.model.add(GRU(self.hidden_len, return_sequences=True,
#                        input_shape=(self.sentence_length,
#                                     self.input_len)))
#     self.model.add(Dropout(dropout))
#
#     for nl in range(nb_layers-1):
#         # check whether return sequences
#         if nl != nb_layers-2:
#             return_sequences_ = True
#         else:
#             return_sequences_ = return_sequences
#         # build hidden layers
#         self.model.add(GRU(self.hidden_len,
#                            return_sequences=return_sequences_))
#         self.model.add(Dropout(dropout))
#
#     if mapping == 'o2o':
#         # if mapping is one-to-one
#         self.model.add(Dense(self.output_len))
#     elif mapping == 'm2m':
#         # if mapping is many-to-many
#         self.model.add(TimeDistributedDense(self.output_len))
#
#     self.model.add(Activation('softmax'))
#
#     self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
