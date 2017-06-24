#include "test.h"

test_nn::test_nn(const char * _dataset, const mat & train_in, const mat & train_out, const mat & test_in, const mat & test_out, const vector<size_t> & n_neurons, const size_t & max_n_iter)
{
	dataset = _dataset;
	h_layers = n_neurons.size();
	string h_units = "";
	for(const size_t hu: n_neurons)
		h_units += " " + to_string(hu);
	
	char iter_file[128];
	auto gen_plot = [&]()
	{
		sprintf(iter_file, "mv tmp/iter_loss tmp/%s_%s.error", dataset, train_type);
		int s = system(iter_file);
	};

	network net(n_neurons.size());

	train_type = "normal";
	TIC(train_time)
	n_iter = net.train(train_in, train_out, n_neurons, max_n_iter);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();
	
	train_type = "momentum";
	TIC(train_time) 
	n_iter = net.train_momentum(train_in, train_out, n_neurons, max_n_iter);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();
	
	train_type = "mini_batches";
	TIC(train_time) 
	n_iter = net.train_sgd(train_in, train_out, n_neurons, max_n_iter, 32);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();

	train_type = "normal_new";
	TIC(train_time)
	n_iter = net.train_new(train_in, train_out, n_neurons, max_n_iter);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();
	
	train_type = "momentum_new";
	TIC(train_time) 
	n_iter = net.train_new(train_in, train_out, n_neurons, max_n_iter, 1, 0.2);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();
	
	train_type = "mini_batches_new";
	TIC(train_time) 
	n_iter = net.train_new(train_in, train_out, n_neurons, max_n_iter, 32);
	TOC(train_time)

	train_error = net.test(train_in, train_out);
	test_error = net.test(test_in, test_out);
	
	PRINT_RESULT gen_plot();

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

