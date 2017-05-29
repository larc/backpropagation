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
	mat inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1}};
	inputs = inputs.t();
	mat outputs = {1, 1, 1, 0};
	
	network net(1);
	net.train(inputs, outputs, 2, 1000);
	cout << "error: " << net.test(inputs, outputs) << endl;
}

void main_iris()
{
	string data = "iris.txt";
	mat inputs(4, 150);
	mat outputs(3, 150, fill::zeros);
	
	string slabel;

	ifstream is(data);
	
	for(index_t i = 0; i < inputs.n_cols; i++)
	{
		is >> inputs(0, i) >> inputs(1, i) >> inputs(2, i) >> inputs(3, i) >> slabel;
		outputs(i / 50, i) = 1;
	}	

	is.close();
	for(index_t i = 0; i < inputs.n_rows; i++)
		inputs.row(i) /= max(inputs.row(i));

	network net(1);
	net.train(inputs, outputs, 6, 1000);
	cout << "error: " << net.test(inputs, outputs) << endl;
}

