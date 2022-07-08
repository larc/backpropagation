# backpropagation

This code is a simple implementation of the backpropagation neural networks algorithm in c++ using the library [Armadillo](http://arma.sourceforge.net/) for matrix operations.

## Algorithm

### Neural network

### $$a^L(x, w^1, ..., w^L) = h^L(h^{L-1}(h^{L-2}(...h^1(x, w^1), ...), w^{L-2}), w^{L-1}), w^L)$$

where $x$: input, $w^l$: layer parameters, $a^l = h^l(x, w^l)$: non-(linear) function.

Given a training set $(X, Y)$ we want to find the optimal parameteres:

### $$w^* = \arg \min_w \sum_{(x, y) \in (X, Y)} L(y, a^L(x, w))$$

### Gradient-based learning $$w^{t+1} = w^t - \eta_t \nabla_w L$$

### Stochastic Gradient Descent $$w^{t + 1} = w^t - \frac{\eta_t}{|B|} \sum_{b \in B} \nabla_w L_b$$

### SGD Momentum $$\Delta w = \alpha \Delta w - \frac{\eta_t}{|B|} \sum_{b \in B} \nabla_w L_b$$
### $$w^{t+1} = w^t + \Delta w$$


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


### new

|      dataset |               train_type |   train_time |       n_iter |     h_layers |  train_error |   test_error |          h_units |
| -----------: | -----------------------: | -----------: | -----------: | -----------: | -----------: | -----------: | ---------------: |
|        mnist |           sigmoid_sgd_32 |     2287.099 |          100 |            1 |        2.745 |        3.350 |              300 |
|         iris |           sigmoid_sgd_32 |        3.457 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |      sigmoid_momentum_32 |        3.418 |        20000 |            2 |        1.667 |        0.000 |              8 6 |
|         iris |              relu_sgd_32 |        1.446 |        10821 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |         relu_momentum_32 |        2.943 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |           sigmoid_sgd_16 |        3.432 |        20000 |            2 |        6.667 |        3.333 |              8 6 |
|         iris |      sigmoid_momentum_16 |        1.030 |         6023 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |              relu_sgd_16 |        0.956 |         7107 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |         relu_momentum_16 |        3.240 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |            sigmoid_sgd_4 |        2.643 |        15035 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_4 |        0.884 |         5013 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |               relu_sgd_4 |        0.680 |         4883 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |          relu_momentum_4 |        4.861 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |            sigmoid_sgd_1 |        3.828 |        20000 |            2 |        2.500 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_1 |        0.653 |         3428 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |               relu_sgd_1 |        0.471 |         3067 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |          relu_momentum_1 |       11.521 |        20000 |            2 |       66.667 |       66.667 |              8 6 |


