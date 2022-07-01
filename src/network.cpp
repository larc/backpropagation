#include "network.h"

#include <cassert>

percent_t network::tol_error = 0.01;
percent_t network::threshold = 0.01;

network::network(const size_t & h_layers)
{
	n_layers = h_layers + 1;
	layers = new layer[n_layers];
}

network::~network()
{
	delete [] layers;
}

const vec & network::o_layer() const
{
	return layers[n_layers - 1];
}

size_t network::train_new(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & max_iter, const size_t & batch_size, const percent_t & alpha)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	deltas_w = new mat[n_layers];
	deltas_b = new vec[n_layers];

	for(index_t i = 0; i < n_layers; i++)
	{
		deltas_w[i].zeros(size(layers[i].weights));
		deltas_b[i].zeros(size(layers[i].bias));
	}

	size_t n_batches = (inputs.n_cols + batch_size - 1) / batch_size;

	size_t c_end, c_start;
	percent_t error = 1;
	size_t iter = 0;
	while(iter++ < max_iter && error > tol_error)
	{
		error = 0;
		c_start = 0;
		for(index_t b = 0; b < n_batches; b++)
		{
			c_end = c_start + batch_size - 1;
			if(c_end >= inputs.n_cols) c_end = inputs.n_cols - 1;

			error += train_batch(inputs.cols(c_start, c_end), outputs.cols(c_start, c_end), alpha);

			c_start += batch_size;
		}
		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	delete [] deltas_w;
	delete [] deltas_b;

	return iter - 1;
}

size_t network::train_sgd(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const size_t & batch_size)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	mat * deltas_w = new mat[n_layers];

	size_t n_batches = (inputs.n_cols + batch_size - 1) / batch_size;

	size_t c_end, c_start;
	percent_t error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(index_t b = 0; b < n_batches; b++)
		{
			for(index_t i = 0; i < n_layers; i++)
				deltas_w[i].zeros(size(layers[i].weights));

			c_start = b * batch_size;
			c_end = b < n_batches - 1 ? c_start + batch_size : inputs.n_cols;

			for(index_t c = c_start; c < c_end; c++)
			{
				const vec & input = inputs.col(c);
				const vec & output = outputs.col(c);

				forward(input, output);
				error += loss > threshold;

				vec dl_da = o_layer() - output;
				mat da_dx;

				for(index_t i = n_layers - 1; i > 0; i--)
					layers[i].backprogation_sgd(dl_da, da_dx, deltas_w[i], layers[i - 1]);

				layers[0].backprogation_sgd(dl_da, da_dx, deltas_w[0], input);

			}

			c_end -= c_start;
			for(index_t i = 0; i < n_layers; i++)
			{
				deltas_w[i] /= c_end;
				layers[i].weights -= eta * deltas_w[i];
			}
		}
		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	delete [] deltas_w;
	return iter;
}

size_t network::train_momentum(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const percent_t & alpha)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	mat * deltas_w = new mat[n_layers];
	for(index_t i = 0; i < n_layers; i++)
		deltas_w[i].zeros(size(layers[i].weights));

	percent_t error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(index_t c = 0; c < inputs.n_cols; c++)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);
			error += loss > threshold;

			vec dl_da = o_layer() - output;
			mat da_dx;

			for(index_t i = n_layers - 1; i > 0; i--)
				layers[i].backprogation_momentum(dl_da, da_dx, deltas_w[i], layers[i - 1], alpha);

			layers[0].backprogation_momentum(dl_da, da_dx, deltas_w[0], input, alpha);
		}

		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	delete [] deltas_w;
	return iter;
}

size_t network::train(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	percent_t error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(index_t c = 0; c < inputs.n_cols; c++)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);
			error += loss > threshold;

			vec dl_da = o_layer() - output;
			mat da_dx;

			for(index_t i = n_layers - 1; i > 0; i--)
				layers[i].backprogation(dl_da, da_dx, layers[i - 1]);

			layers[0].backprogation(dl_da, da_dx, input);
		}

		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	return iter;
}

percent_t network::test(const mat & inputs, const mat & outputs)
{
	percent_t error = 0;
	for(index_t c = 0; c < inputs.n_cols; c++)
	{
		forward(inputs.col(c), outputs.col(c));
	//	cout << outputs.col(c).t() << endl;
	//	cout << o_layer().t() << endl;
	//	cout << loss << endl;
		error += loss > threshold;
	}

	return error / inputs.n_cols;
}

void network::init(const size_t & size_in, const size_t & size_out, const vector<size_t> & n_neurons)
{
	assert(n_neurons.size() == n_layers - 1);

//	arma_rng::set_seed_random();

	layers[0].init(size_in, n_neurons.front());
	for(index_t i = 1; i < n_neurons.size(); i++)
		layers[i].init(n_neurons[i - 1], n_neurons[i]);
	layers[n_layers - 1].init(n_neurons.back(), size_out);
}

void network::forward(const vec & input, const vec & output)
{
	layers[0].forward(input);
	for(index_t i = 1; i < n_layers; i++)
		layers[i].forward(layers[i-1]);

	loss = norm(output - o_layer());
	loss = 0.5 * loss * loss;
}

percent_t network::train_batch(const mat & inputs, const mat & outputs, const percent_t & alpha)
{
	percent_t error = 0;
	const size_t & n_inputs = inputs.n_cols;

	for(index_t l = 0; l < n_layers; l++)
	{
		deltas_w[l] *= alpha;
		deltas_b[l] *= alpha;
	}

	percent_t beta = ((1 - alpha) * (- eta)) / n_inputs;

	for(index_t i = 0; i < n_inputs; i++)
	{
		const vec & input = inputs.col(i);
		const vec & output = outputs.col(i);

		forward(input, output);
		error += loss > threshold;

		vec dl_da = o_layer() - output;
		mat da_dx, dl_dw;

		for(index_t l = n_layers - 1; l > 0; l--)
		{
//			layers[l].backprogation(dl_da, da_dx, layers[l - 1]);
			layers[l].compute_gradients(dl_da, da_dx, dl_dw, layers[l - 1]);
			layers[l].weights += beta * dl_dw;
			layers[l].bias = beta * dl_da;
//			deltas_w[l] += beta * dl_dw;
//			deltas_b[l] += beta * dl_da;
		}
//		layers[0].backprogation(dl_da, da_dx, input);
		layers[0].compute_gradients(dl_da, da_dx, dl_dw, input);
		layers[0].weights += beta * dl_dw;
		layers[0].bias = beta * dl_da;
//		deltas_w[0] += beta * dl_dw;
//		deltas_b[0] += beta * dl_da;
	}
/*
	for(index_t l = 0; l < n_layers; l++)
	{
		layers[l].weights += deltas_w[l];
		layers[l].bias = deltas_b[l];
	}*/

	return error;
}

