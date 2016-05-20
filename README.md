# sequence-rnn-py

[![Build Status](https://travis-ci.org/fluency03/sequence-rnn-py.svg?branch=master)](https://travis-ci.org/fluency03/sequence-rnn-py)

This program analyze the sequence using (Uni-directional and Bi-directional) Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) based on the python library Keras ([Documents](http://keras.io/) and [Github](https://github.com/fchollet/keras)).
It is based on this [lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) and this [imdb_bidirectional_lstm.py]( https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py) examples of Keras.


*This is part of my master thesis project and still in development.*

## Requirements

- [Python 2.7](https://www.python.org/downloads/)
- [NumPy](http://www.numpy.org/): The fundamental package needed for scientific computing with Python.
- [SciPy](http://scipy.org/):  Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [Theano](http://deeplearning.net/software/theano/): A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
- [Tensorflow](https://www.tensorflow.org/): An open source software library for numerical computation using data flow graphs.
- [Keras>=1.0](http://keras.io/): A minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. Update the Keras:

    `pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps` .

- **GPU Support** (optional but highly recommended). Instructions of enabling GPU are here: [for Theano](http://deeplearning.net/software/theano/install.html#using-the-gpu) and [for TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#optional-linux-enable-gpu-support).
- [pydot](https://github.com/erocarrera/pydot) and [graphviz](http://www.graphviz.org/) (optional, if you want to plot the model)
- [HDF5](https://www.hdfgroup.org/HDF5/) and [h5py](http://www.h5py.org/) (optional, if you use model saving/loading functions)


## Materials

A serias of Recurrent Neural Networks Tutorial:

1. [Part 1 - Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
2. [Part 2 - Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
3. [Part 3 - Backpropagation Through Time and Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
4. [Part 4 - Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

Two great materials about LSTM: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) of [Christopher Olah](http://colah.github.io/) and [Understanding LSTM and its diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714#.5hkwmotmr) of [Shi Yan](https://medium.com/@shiyan)

The best post of [Andrej Karpathy blog](http://karpathy.github.io/) regarding sequence prediction using RNN: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

One deeper material about RNN: [Chapter 10 - Sequence Modeling: Recurrentand Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html) of this book [MIT Deep Learning](http://www.deeplearningbook.org/).


## Model


- Two layers of LSTMs Uni-directional RNN model:

![ RNN LSTM ](https://github.com/fluency03/sequence-rnn-py/blob/master/rnn_model.png "RNN LSTM")


- One layer of LSTM Bi-Directional RNN model:

![ BRNN LSTM ](https://github.com/fluency03/sequence-rnn-py/blob/master/brnn_model.png "BRNN LSTM")


- Naive Bayes model:

[naive_bayes.py](https://github.com/fluency03/sequence-rnn-py/blob/master/naive_bayes.py)

## Data

- Training Set

- Validation Set

- Test Set



## Training

This [hyperas](https://github.com/maxpumperla/hyperas) may help. It is *A very simple convenience wrapper around [hyperopt](https://github.com/hyperopt/hyperopt) for fast prototyping with keras models.* It is used for hyper-parameter optimization. An example can be found [here](https://github.com/maxpumperla/hyperas/blob/master/examples/lstm.py).

Two good materials:

- [CHAPTER 3: Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html) from [Michael Nielsen](http://michaelnielsen.org/)
- [Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/) and [Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/) from Stanford CS class [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)

Considerations:

- **Batch Size**: how many streams of data are processed in parallel at one time.


- **Samples per epoch** and **Batches per epoch**: how many samples or batches considered per epoch. Based on some of my experiments: (i) the more #samples there are, the higher the accuracy can reach at the stable stage and the less the loss can be at the stable stage; (ii) the more #batches (integer ratio of #sample/batch_size) there are, the higher the accuracy can reach at table stage and the less the loss can be at stable stage and the less iterations it will take to reach the same loss/accuracy value.


- **Sentence Length**: according to [char-rnn](https://github.com/karpathy/char-rnn):
 > The length of each data stream, which is also the limit at which the gradients can propagate backwards in time. For example, if seq_length is 20, then the gradient signal will never backpropagate more than 20 time steps, and the model might not find dependencies longer than this length in number of characters.

 This is actually the limitation of the model's long term memory.

 > Thus, if you have a very difficult dataset where there are a lot of long-term dependencies, you will want to increase this setting.


- **Offset during sampling**: offset is the start index when sampling the X_train and y_train from original sequence. The offset can be fixed value or random value ranging between 0 ~ step-1.


- **Data size vs. #parameters** in total:
 - #layers: the number of layers, [here](https://github.com/karpathy/char-rnn) suggests that always use num_layers of either 2 or 3.
 - layer size: the number of units per layer.

 Acoording to [char-rnn](https://github.com/karpathy/char-rnn), the two important quantities to keep track of here are:
 - The total number of parameters in your model.
 - The size of your dataset.
 These two should be about the same order of magnitude.

 **How to calculate the number of parameters in RNN?** For example, consider one layer of LSTM:
 - if it has the layer size of `H=512`;
 - if we have the vocabulary size as `C=3000` (the number of unique classes);
 - the LSTM layer will have three parameter matrix - `U` with dimension `(H, C)=(512, 3000)`, `V` with dimension `(C, H)=(3000, 512)`, `W` with dimension `(H, H)=(512, 512)`;
 - the total number of parameter for one layer will be: `2HC + H^2`, which is **3,334,144** in this case.
 - That is 3 million parameters for only one layer!


- **Learning Rate**: This ratio (percentage) influences the speed (step of the gradient descent) and quality of learning. The greater the ratio, the faster the neuron trains; the lower the ratio, the more accurate the training is. According to [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf) [\[1\]](https://github.com/fluency03/sequence-rnn-py#1-greff-klaus-rupesh-kumar-srivastava-jan-koutník-bas-r-steunebrink-and-jürgen-schmidhuber-lstm-a-search-space-odyssey-arxiv-preprint-arxiv150304069-2015):
> The learning rate is by far the most important hyperparameter. And based on their suggestion, while searching for a good learning rate for the LSTM, it is sufficient to do a coarse search by starting with a high value (e.g. 1.0) and dividing it by ten until performance stops increasing.


- **[Dropout](http://keras.io/layers/core/#dropout)**: an float between 0 and 1, indicating how much percentage of the hidden layer data are ignored when feeding to next layer. It is a powerful regularization method and mainly used for avoiding overfitting. If your model is overfitting, it better to increase the value of dropout.


- **Reinforcement learning function**: The *temperature* parameter is dividing the predicted log probabilities before the *[Softmax](https://en.wikipedia.org/wiki/Softmax_function)*, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.


- **Loss function**: [categorical_crossentropy](http://keras.io/objectives/)


- **Optimizer**: [RMSprop](http://keras.io/optimizers/#rmsprop), you can try other options like simple [SGD](http://keras.io/optimizers/#sgd), [Adagrad](http://keras.io/optimizers/#adagrad) and [Adam](http://keras.io/optimizers/#adam).


## Reference

###### [1] Greff, Klaus, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber. "*[LSTM: A search space odyssey.](http://arxiv.org/pdf/1503.04069v1.pdf)*" arXiv preprint arXiv:1503.04069 (2015).
