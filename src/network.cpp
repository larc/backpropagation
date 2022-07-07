#include "network.h"

#include <cassert>


double network::tol_error = 0.01;
double network::threshold = 0.01;

network::network(const size_t & h_layers)
{
	layers.resize(h_layers + 1);
}

const vec & network::o_layer() const
{
	return layers.back();
}

size_t network::train_new(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & max_iter, const size_t & batch_size, const double & alpha)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	deltas_w.resize(layers.size());
	deltas_b.resize(layers.size());

	for(uint_t i = 0; i < layers.size(); ++i)
	{
		deltas_w[i].zeros(size(layers[i].weights));
		deltas_b[i].zeros(size(layers[i].bias));
	}

	size_t n_batches = (inputs.n_cols + batch_size - 1) / batch_size;

	size_t c_end, c_start;
	double error = 1;
	size_t iter = 0;
	while(iter++ < max_iter && error > tol_error)
	{
		error = 0;
		c_start = 0;
		for(uint_t b = 0; b < n_batches; ++b)
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

	return iter - 1;
}

size_t network::train_sgd(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const size_t & batch_size)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	deltas_w.resize(layers.size());

	size_t n_batches = (inputs.n_cols + batch_size - 1) / batch_size;

	size_t c_end, c_start;
	double error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(uint_t b = 0; b < n_batches; ++b)
		{
			for(uint_t i = 0; i < layers.size(); ++i)
				deltas_w[i].zeros(size(layers[i].weights));

			c_start = b * batch_size;
			c_end = b < n_batches - 1 ? c_start + batch_size : inputs.n_cols;

			for(uint_t c = c_start; c < c_end; ++c)
			{
				const vec & input = inputs.col(c);
				const vec & output = outputs.col(c);

				forward(input, output);
				error += loss > threshold;

				vec dl_da = o_layer() - output;
				mat da_dx;

				for(uint_t i = layers.size() - 1; i > 0; i--)
					layers[i].backward_sgd(dl_da, da_dx, deltas_w[i], layers[i - 1]);

				layers[0].backward_sgd(dl_da, da_dx, deltas_w[0], input);

			}

			c_end -= c_start;
			for(uint_t i = 0; i < layers.size(); ++i)
			{
				deltas_w[i] /= c_end;
				layers[i].weights -= eta * deltas_w[i];
			}
		}
		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	return iter;
}

size_t network::train_momentum(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const double & alpha)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	deltas_w.resize(layers.size());
	for(uint_t i = 0; i < layers.size(); ++i)
		deltas_w[i].zeros(size(layers[i].weights));

	double error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(uint_t c = 0; c < inputs.n_cols; ++c)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);
			error += loss > threshold;

			vec dl_da = o_layer() - output;
			mat da_dx;

			for(uint_t i = layers.size() - 1; i > 0; i--)
				layers[i].backward_momentum(dl_da, da_dx, deltas_w[i], layers[i - 1], alpha);

			layers[0].backward_momentum(dl_da, da_dx, deltas_w[0], input, alpha);
		}

		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	return iter;
}

size_t network::train(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter)
{
	_os_open
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	double error = 1;
	size_t iter = 0;
	while(iter < n_iter && error > tol_error)
	{
		iter++;

		error = 0;
		for(uint_t c = 0; c < inputs.n_cols; ++c)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);
			error += loss > threshold;

			vec dl_da = o_layer() - output;
			mat da_dx;

			for(uint_t i = layers.size() - 1; i > 0; i--)
				layers[i].backward(dl_da, da_dx, layers[i - 1]);

			layers[0].backward(dl_da, da_dx, input);
		}

		error /= inputs.n_cols;
		_os_error(iter, error)
	}
	_os_close

	return iter;
}

double network::test(const mat & inputs, const mat & outputs)
{
	double error = 0;
	for(uint_t c = 0; c < inputs.n_cols; ++c)
	{
		forward(inputs.col(c), outputs.col(c));
		error += loss > threshold;
	}

	return error / inputs.n_cols;
}

void network::init(const size_t & size_in, const size_t & size_out, const vector<size_t> & n_neurons)
{
	assert(n_neurons.size() == layers.size() - 1);

//	arma_rng::set_seed_random();

	layers[0].init(size_in, n_neurons.front());
	for(uint_t i = 1; i < n_neurons.size(); ++i)
		layers[i].init(n_neurons[i - 1], n_neurons[i]);
	layers[layers.size() - 1].init(n_neurons.back(), size_out);
}

void network::forward(const vec & input, const vec & output)
{
	layers[0].forward(input);
	for(uint_t i = 1; i < layers.size(); ++i)
		layers[i].forward(layers[i-1]);

	loss = norm(output - o_layer());
	loss = 0.5 * loss * loss;
}

double network::train_batch(const mat & inputs, const mat & outputs, const double & alpha)
{
	double error = 0;
	const size_t & n_inputs = inputs.n_cols;

	for(uint_t l = 0; l < layers.size(); ++l)
	{
		deltas_w[l] *= alpha;
		deltas_b[l] *= alpha;
	}

	double beta = ((1 - alpha) * (- eta)) / n_inputs;

	for(uint_t i = 0; i < n_inputs; ++i)
	{
		const vec & input = inputs.col(i);
		const vec & output = outputs.col(i);

		forward(input, output);
		error += loss > threshold;

		vec dl_da = o_layer() - output;
		mat da_dx, dl_dw;

		for(uint_t l = layers.size() - 1; l > 0; l--)
		{
//			layers[l].backward(dl_da, da_dx, layers[l - 1]);
			layers[l].compute_gradients(dl_da, da_dx, dl_dw, layers[l - 1]);
			layers[l].weights += beta * dl_dw;
			layers[l].bias = beta * dl_da;
//			deltas_w[l] += beta * dl_dw;
//			deltas_b[l] += beta * dl_da;
		}
//		layers[0].backward(dl_da, da_dx, input);
		layers[0].compute_gradients(dl_da, da_dx, dl_dw, input);
		layers[0].weights += beta * dl_dw;
		layers[0].bias = beta * dl_da;
//		deltas_w[0] += beta * dl_dw;
//		deltas_b[0] += beta * dl_da;
	}
/*
	for(uint_t l = 0; l < layers.size(); ++l)
	{
		layers[l].weights += deltas_w[l];
		layers[l].bias = deltas_b[l];
	}*/

	return error;
}

