# RNN-for-image-captioning
This code is to build RNN and LSTM model for image captioning from scratch.

## Getting started
This code is written in python and the libraries needed are listed in the file requirements.txt.

## Theory background
This code implements two kinds of models: recurrent neural network (RNN) and long short term memory (LSTM). The foundation theory for RNN can be refered to [this RNN chapter](http://www.deeplearningbook.org/contents/rnn.html#pf9) and [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and that for LSTM can be refered to [Christopher Olah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

## Framework
This code follows a modular design. The training process is divided into two steps. First of all, a RNN or LSTM model is constructed, where the network structure is detailedly defined and the forward and backward flow is implemented. Secondly, a certain kind of solver is called to update the learnable parameters of the model built previously and output the evaluated loss results periodically. There are four kinds of first-order update methods available: Vanilla SGD, SGD Momentum, RMSprop and Adam.

## Demo
To illustrate how to use this code, one demo sample is provided. Specifically, both RNN and LSTM models (one layer) are individually constructed to train on the MS-COCO dataset and ouput the image captions for test images. You will need to download the MS-COCO dataset. To do so, please run the following command from the Dataset directory:

./get_coco_captioning.sh

## Author
- **Di Jin** -- jindi15@mit.edu

## Acknowledgement
This code is based on the [assignment 3](http://cs231n.github.io/assignments2016/assignment3/) of [Stanford cs231n course](http://cs231n.stanford.edu/). Thanks a lot for this outstanding course!
