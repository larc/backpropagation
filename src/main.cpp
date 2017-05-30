#include "network.h"

#include <iostream>
#include <fstream>

using namespace std;

void main_iris();
void main_xor();

int main()
{
	main_iris();
	main_xor();

	return 0;
}

void main_xor()
{
	mat inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	inputs = inputs.t();
	mat outputs = {1, 1, 1, 0};
	
	network net(1);
	net.train(inputs, outputs, {2}, 1000);
	cout << "error: " << net.test(inputs, outputs) << endl;
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

	network net(2);
	net.train(train_in, train_out, {8, 6}, 6000);
	cout << "error train: " << net.test(train_in, train_out) << endl;
	cout << "error test: " << net.test(test_in, test_out) << endl;
}

