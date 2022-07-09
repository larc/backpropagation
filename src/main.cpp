#include "test.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>

using namespace std;


void main_iris();
void main_mnist();

int main()
{
	srand(time(NULL));

	PRINT_HEADER
//	main_mnist();
	main_iris();

	return 0;
}

void main_mnist()
{
	mat train_in, test_in;
	mat train_out, test_out;

	read_mnist_labels(train_out, data_path("train-labels.idx1-ubyte"));
	read_mnist_data(train_in, data_path("train-images.idx3-ubyte"));

	read_mnist_labels(test_out, data_path("t10k-labels.idx1-ubyte"));
	read_mnist_data(test_in, data_path("t10k-images.idx3-ubyte"));

	train_in /= 255;
	test_in /= 255;

	// http://yann.lecun.com/exdb/mnist/
	test_nn("mnist", train_in, train_out, test_in, test_out, {300}, 1000, 0.001);
	test_nn("mnist", train_in, train_out, test_in, test_out, {300, 100}, 1000, 0.001);
	test_nn("mnist", train_in, train_out, test_in, test_out, {500, 150}, 1000, 0.001);
}


void main_iris()
{
	string slabel;
	ifstream is(data_path("iris.txt"));

	mat data(4, 150);
	vector<uint_t> labels(150);
	for(uint_t i = 0; i < data.n_cols; ++i)
	{
		is >> data(0, i) >> data(1, i) >> data(2, i) >> data(3, i) >> slabel;
		labels[i] = i / 50;
	}

	is.close();


	mat train_in(4, 120);
	mat train_out(3, 120, fill::zeros);
	mat test_in(4, 30);
	mat test_out(3, 30, fill::zeros);

	vector<bool> visited;
	visited.assign(data.n_cols, false);

	uint_t ix;
	for(uint_t i = 0; i < train_in.n_cols; ++i)
	{
		while(visited[ix = rand() % data.n_cols]);

		visited[ix] = true;
		train_in.col(i) = data.col(ix);
		train_out(labels[ix], i) = 1;
	}

	for(uint_t i = 0; i < test_in.n_cols; ++i)
	{
		while(visited[ix = rand() % data.n_cols]);

		visited[ix] = true;
		test_in.col(i) = data.col(ix);
		test_out(labels[ix], i) = 1;
	}


	auto normalise = [](mat & in)
	{
		double max_v, min_v;
		for(uint_t i = 0; i < in.n_rows; ++i)
		{
			max_v = max(in.row(i));
			min_v = min(in.row(i));
			in.row(i) = (in.row(i) - min_v) / (max_v - min_v);
		}
	};

	normalise(train_in);
	normalise(test_in);

	test_nn("iris", train_in, train_out, test_in, test_out, {8, 6}, 100000, 0.1);
}

