#include "test.h"

#include <cassert>


test_nn::test_nn(const char * _dataset, const mat & train_in, const mat & train_out, const mat & test_in, const mat & test_out, const vector<size_t> & n_neurons, const size_t & max_n_iter, const double & eta)
{
	dataset = _dataset;
	h_layers = n_neurons.size();
	string h_units = "";
	for(const size_t hu: n_neurons)
		h_units += " " + to_string(hu);

	char iter_file[128];
	auto gen_plot = [&]()
	{
		sprintf(iter_file, "mv %s/tmp/iter_loss %s/tmp/%s_%s.error", SRC_PATH, SRC_PATH, dataset, train_type);
		system(iter_file);
	};

	network net(n_neurons.size());
	net.eta = eta;

	auto test = [&](const char * type, const size_t & batch_size, const double & alpha_momentum)
	{
		train_type = type;
		TIC(train_time)
		n_iter = net.train(train_in, train_out, n_neurons, max_n_iter, batch_size, alpha_momentum);
		TOC(train_time)

		train_error = net.test(train_in, train_out);
		test_error = net.test(test_in, test_out);

		PRINT_RESULT gen_plot();
	};

	char train_label[128];
	for(uint_t b: {32, 16, 4, 1})
	{
		layer::f = sigmoid;
		layer::df = d_sigmoid;

		sprintf(train_label, "sigmoid_sgd_%d", b);
		test(train_label, b, 0);

		sprintf(train_label, "sigmoid_momentum_%d", b);
		test(train_label, b, 0.9);

		//layer::f = relu;
		//layer::df = d_relu;

		//sprintf(train_label, "relu_sgd_%d", b);
		//test(train_label, b, 0);

		//sprintf(train_label, "relu_momentum_%d", b);
		//test(train_label, b, 0.9);
	}
}

/**************************************************************************************************/

void read_mnist_labels(mat & labels, const string & file)
{
	ifstream is(file, ios::binary);
	assert(is);

	int32_t m, n;
	is.read((char *) &m, sizeof(int32_t));
	is.read((char *) &n, sizeof(int32_t));
	m = __builtin_bswap32(m);
	n = __builtin_bswap32(n);

	uword size = 10;
	labels.resize(size, n);
	labels.zeros();

	int8_t label;
	for(int32_t i = 0; i < n; ++i)
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

	for(int32_t i = 0; i < n; ++i)
	{
		is.read((char *) buffer, size);
		memptr = data.colptr(i);
		for(uint_t j = 0; j < size; ++j)
			memptr[j] = (double) buffer[j];
	}

	is.close();
}

