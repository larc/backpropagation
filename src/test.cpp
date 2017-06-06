#include "test.h"

test_nn::test_nn(const char * _dataset, const mat & train_in, const mat & train_out, const mat & test_in, const mat & test_out, const vector<size_t> & n_neurons, const size_t & _n_iter)
{
	dataset = _dataset;
	n_iter = _n_iter;
	h_layers = n_neurons.size();

	network net(n_neurons.size());
	TIC(train_time) net.train(train_in, train_out, n_neurons, n_iter); TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT
}

/**************************************************************************************************/

void read_mnist_labels(mat & labels, const string & file)
{	
	ifstream is(file, ios::binary);

	int32_t m, n;
	is.read((char *) &m, sizeof(int32_t));
	is.read((char *) &n, sizeof(int32_t));
	m = __builtin_bswap32(m);
	n = __builtin_bswap32(n);

	uword size = 10;
	labels.resize(size, n);
	labels.zeros();
	
	int8_t label;
	for(index_t i = 0; i < n; i++)
	{
		is.read((char *) &label, sizeof(int8_t));
		labels((int) label, i) = 1;
	}
	
	is.close();
}

void read_mnist_data(mat & data, const string & file)
{	
	ifstream is(file, ios::binary);

	int32_t m, n, r, c;
	is.read((char *) &m, sizeof(int32_t));
	is.read((char *) &n, sizeof(int32_t));
	is.read((char *) &r, sizeof(int32_t));
	is.read((char *) &c, sizeof(int32_t));
	m = __builtin_bswap32(m);
	n = __builtin_bswap32(n);
	r = __builtin_bswap32(r);
	c = __builtin_bswap32(c);
	
	size_t size = r * c;
	unsigned char buffer[size];

	data.resize(size, n);
	double * memptr;

	for(index_t i = 0; i < n; i++)
	{
		is.read((char *) buffer, size);
		memptr = data.colptr(i);
		for(index_t j = 0; j < size; j++)
			memptr[j] = (double) buffer[j];
	}

	is.close();
}

