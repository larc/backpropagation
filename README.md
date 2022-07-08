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
|         iris |           sigmoid_sgd_32 |        3.509 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |      sigmoid_momentum_32 |        3.445 |        20000 |            2 |       66.667 |       66.667 |              8 6 |
|         iris |      relu_sigmoid_sgd_32 |        1.444 |        10640 |            2 |        0.833 |        0.000 |              8 6 |
|         iris | relu_sigmoid_momentum_32 |        1.537 |        11307 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |           sigmoid_sgd_16 |        3.517 |        20000 |            2 |        8.333 |       13.333 |              8 6 |
|         iris |      sigmoid_momentum_16 |        3.539 |        20000 |            2 |       28.333 |       26.667 |              8 6 |
|         iris |      relu_sigmoid_sgd_16 |        1.102 |         7993 |            2 |        0.833 |        0.000 |              8 6 |
|         iris | relu_sigmoid_momentum_16 |        0.969 |         7172 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |            sigmoid_sgd_4 |        2.710 |        14907 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_4 |        2.797 |        15376 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |       relu_sigmoid_sgd_4 |        0.787 |         5540 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |  relu_sigmoid_momentum_4 |        0.758 |         5248 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |            sigmoid_sgd_1 |        1.645 |         8245 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |       sigmoid_momentum_1 |        1.806 |         8926 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |       relu_sigmoid_sgd_1 |        0.523 |         3224 |            2 |        0.833 |        0.000 |              8 6 |
|         iris |  relu_sigmoid_momentum_1 |        0.465 |         2822 |            2 |        0.833 |        0.000 |              8 6 |
