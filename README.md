# backpropagation

This code is a simple implementation of the backpropagation neural networks algorithm in c++ using the library [Armadillo](http://arma.sourceforge.net/) for matrix operations.

## Build and Compile

    mkdir build
    cd build
    cmake ..
    make
    ./backprogation

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

