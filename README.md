# backpropagation

This code is a simple implementation of the backpropagation neural networks algorithm in c++ using the library [Armadillo](http://arma.sourceforge.net/) for matrix operations.

## Algorithm

### Neural network

### $$a^L(x, w^1, ..., w^L) = h^L(h^{L-1}(h^{L-2}(...h^1(x, w^1), ...), w^{L-2}), w^{L-1}), w^L)$$

where $x$: input, $w^l$: layer parameters, $a^l = h^l(x, w^l)$ non-(linear) function.

Given a training set $(X, Y)$ we want to find the optimal parameteres:

### $$w^* = \arg \min_w \sum_{(x, y) \in (X, Y)} L(y, a^L(x, w))$$

### Gradient-based learning

### $$w^{t+1} = w^t - \eta_t \nabla_w L$$

## Build and Compile

	mkdir build
	cd build
	cmake ..
	make
	./backpropagation

## Datasets

### mnist
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### iris
[https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris)

## Test and Results

The test folder contains some scripts to plot the curve of error per iteration. The table below shows the output with some results for training and testing with the mnist and iris datasets.

|      dataset |       train_type |   train_time |       n_iter |     h_layers |  train_error |   test_error |          h_units |
| -----------: | ---------------: | -----------: | -----------: | -----------: | -----------: | -----------: | ---------------: |
|        mnist |           normal |     1122.792 |           45 |            1 |        1.593 |        6.860 |              300 |
|        mnist |         momentum |     1574.166 |           46 |            1 |        1.277 |        6.860 |              300 |
|        mnist |     mini_batches |     1251.950 |          100 |            1 |       17.890 |       18.200 |              300 |
|        mnist |       normal_new |     1469.569 |           47 |            1 |        1.572 |        6.830 |              300 |
|        mnist |     momentum_new |     1515.006 |           54 |            1 |        1.392 |        6.730 |              300 |
|        mnist | mini_batches_new |     2378.059 |          100 |            1 |       18.183 |       18.340 |              300 |
|         iris |           normal |        1.650 |        12621 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |         momentum |        1.704 |        12555 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |     mini_batches |        2.365 |        20000 |            2 |        8.333 |        3.333 |              8 6 |
|         iris |       normal_new |        1.643 |        12462 |            2 |        1.667 |       10.000 |              8 6 |
|         iris |     momentum_new |        2.629 |        20000 |            2 |        3.333 |       13.333 |              8 6 |
|         iris | mini_batches_new |        2.536 |        20000 |            2 |      100.000 |      100.000 |              8 6 |

### new

|      dataset |               train_type |   train_time |       n_iter |     h_layers |  train_error |   test_error |          h_units |
| -----------: | -----------------------: | -----------: | -----------: | -----------: | -----------: | -----------: | ---------------: |
|         iris |           sigmoid_sgd_32 |        2.881 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |      sigmoid_momentum_32 |        2.850 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |      relu_sigmoid_sgd_32 |        0.125 |         1078 |            2 |        4.167 |        0.000 |              8 6 |
|         iris | relu_sigmoid_momentum_32 |        0.284 |         2440 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |           sigmoid_sgd_16 |        2.896 |        20000 |            2 |        6.667 |        3.333 |              8 6 |
|         iris |      sigmoid_momentum_16 |        2.911 |        20000 |            2 |        6.667 |        3.333 |              8 6 |
|         iris |      relu_sigmoid_sgd_16 |        0.054 |          450 |            2 |        4.167 |        0.000 |              8 6 |
|         iris | relu_sigmoid_momentum_16 |        0.044 |          367 |            2 |        3.333 |        0.000 |              8 6 |
|         iris |            sigmoid_sgd_4 |        2.219 |        14989 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_4 |        2.125 |        14377 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |       relu_sigmoid_sgd_4 |        0.026 |          212 |            2 |        4.167 |        0.000 |              8 6 |
|         iris |  relu_sigmoid_momentum_4 |        0.012 |           95 |            2 |        7.500 |        0.000 |              8 6 |
|         iris |            sigmoid_sgd_1 |        0.261 |         1580 |            2 |        2.500 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_1 |        2.016 |        12270 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |       relu_sigmoid_sgd_1 |        0.037 |          264 |            2 |        4.167 |        0.000 |              8 6 |
|         iris |  relu_sigmoid_momentum_1 |        0.017 |          124 |            2 |        3.333 |        0.000 |              8 6 |


