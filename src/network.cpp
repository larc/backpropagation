#include "network.h"

#include <cassert>
#include <cstdlib>


network::network(const size_t & h_layers)
{
	srand(time(NULL));
	layers.resize(h_layers + 1);
}

const vec & network::o_layer() const
{
	return layers.back();
}

size_t network::train_sgd(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const size_t & batch_size)
{
	_os_open

	init(inputs.n_rows, outputs.n_rows, n_neurons);

	deltas_w.resize(layers.size());
	std::vector<bool> visited;

	double error = 1;
	size_t iter = 0;
	uint_t ix;

	vec dl_da;
	
	for(uint_t i = 0; i < layers.size(); ++i)
		deltas_w[i].zeros(size(layers[i].weights));

	while(iter < n_iter && error > tol_error)
	{
		++iter;
		visited.assign(inputs.n_cols, false);

		error = 0;
		for(uint_t s = 0; s < inputs.n_cols; ++s)
		{	
			while(visited[ix = rand() % inputs.n_cols]);

			visited[ix] = true;
			const vec & input = inputs.col(ix);
			const vec & output = outputs.col(ix);

			forward(input, output);
			error += o_layer().index_max() != output.index_max();

			vec dl_da = o_layer() - output;
			mat da_dx;
			for(uint_t i = layers.size() - 1; i > 0; --i)
				layers[i].backward_sgd(dl_da, da_dx, deltas_w[i], layers[i - 1]);
			layers[0].backward_sgd(dl_da, da_dx, deltas_w[0], input);
			
			if(!(s % batch_size))
			{
				for(uint_t i = 0; i < layers.size(); ++i)
				{
					layers[i].weights -= eta * deltas_w[i] / batch_size;
					deltas_w[i].zeros();
				}
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
			error += o_layer().index_max() != output.index_max();

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
			error += o_layer().index_max() != output.index_max();

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
	uint_t error = 0;

	for(uint_t c = 0; c < inputs.n_cols; ++c)
	{
		forward(inputs.col(c), outputs.col(c));
		error += o_layer().index_max() != outputs.col(c).index_max();
	}

	return double(error) / inputs.n_cols;
}

void network::init(const size_t & size_in, const size_t & size_out, const vector<size_t> & n_neurons)
{
	assert(n_neurons.size() == layers.size() - 1);

	arma_rng::set_seed_random();

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

