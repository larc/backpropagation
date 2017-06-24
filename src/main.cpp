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
	PRINT_HEADER
//	main_mnist();
	main_iris();

	return 0;
}

void main_mnist()
{
	mat train_in, test_in;
	mat train_out, test_out;
	
	read_mnist_labels(train_out, "tmp/train-labels-idx1-ubyte");
	read_mnist_data(train_in, "tmp/train-images-idx3-ubyte");
	
	read_mnist_labels(test_out, "tmp/t10k-labels-idx1-ubyte");
	read_mnist_data(test_in, "tmp/t10k-images-idx3-ubyte");

	train_in /= 255;
	test_in /= 255;

	test_nn("mnist", train_in, train_out, test_in, test_out, {25}, 21);
	//test_nn("mnist", train_in, train_out, test_in, test_out, {300}, 1);
}


void main_iris()
{
	string data = "iris.txt";
	mat train_in(4, 120);
	mat train_out(3, 120, fill::zeros);
	mat test_in(4, 30);
	mat test_out(3, 30, fill::zeros);
	
	string slabel;

	ifstream is(data);

	for(index_t a = 0, b = 0, i = 0; i < 150; i++)
	{
		if( (i % 50) < 40)
		{
			is >> train_in(0, a) >> train_in(1, a) >> train_in(2, a) >> train_in(3, a) >> slabel;
			train_out(i / 50, a) = 1;
			a++;
		}
		else
		{
			is >> test_in(0, b) >> test_in(1, b) >> test_in(2, b) >> test_in(3, b) >> slabel;
			test_out(i / 50, b) = 1;
			b++;
		}
	}	

	is.close();
	
	auto normalise = [](mat & in)
	{
		double max_v, min_v;
		for(index_t i = 0; i < in.n_rows; i++)
		{
			max_v = max(in.row(i));
			min_v = min(in.row(i));
			in.row(i) = (in.row(i) - min_v) / (max_v - min_v);
		}
	};

	normalise(train_in);
	normalise(test_in);

	shuffle(train_in);
	test_nn("iris", train_in, train_out, test_in, test_out, {8, 6}, 20000);
}

