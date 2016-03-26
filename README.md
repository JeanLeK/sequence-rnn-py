# sequence-rnn-py

[![Build Status](https://travis-ci.org/fluency03/sequence-rnn-py.svg?branch=master)](https://travis-ci.org/fluency03/sequence-rnn-py)

This program analyze the sequence using Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) based on the python library [Keras](http://keras.io/).
It is based on this [lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) text generation exampe of Keras.



## Requirements

- [Python 2.7](https://www.python.org/downloads/)
- [NumPy](http://www.numpy.org/): The fundamental package needed for scientific computing with Python.
- [SciPy](http://scipy.org/):  Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [Theano](http://deeplearning.net/software/theano/): A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
- [Tensorflow](https://www.tensorflow.org/): An open source software library for numerical computation using data flow graphs.
- [Keras](http://keras.io/): A minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano.
- **GPU Support** (Optional but highly recommended). Instructions of enabling GPU are here: [for Theano](http://deeplearning.net/software/theano/install.html#using-the-gpu) and [for TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#optional-linux-enable-gpu-support).



## Materials

A serias of Recurrent Neural Networks Tutorial:

1. [Part 1 - Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
2. [Part 2 - Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
3. [Part 3 - Backpropagation Through Time and Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
4. [Part 4 - Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

Two great materials about LSTM: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of [Christopher Olah](http://colah.github.io/) and [Understanding LSTM and its diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714#.5hkwmotmr) of [Shi Yan](https://medium.com/@shiyan)

The best post of [Andrej Karpathy blog](http://karpathy.github.io/) regarding sequence prediction using RNN: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

One deeper material about RNN: [Chapter 10 - Sequence Modeling: Recurrentand Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html) of this book [MIT Deep Learning](http://www.deeplearningbook.org/).



## Data

- Training Set

- Validation Set (10% of the training data)

- Test Set



## Training (TODO: considerations)

- sentence length
- batch size
- overall data size (-> #hidden layer and size -> #parameters)
- learning rate
- dropout
- reinforcement learning function
- activation
- loss function (objective function)
- optimizer
- regularizer
