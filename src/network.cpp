#include "network.h"

#include <cassert>

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

void network::train_momentum(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, size_t n_iter, const percent_t & alpha)
{
	init(inputs.n_rows, outputs.n_rows, n_neurons);
	
	mat * deltas_w = new mat[n_layers];
	for(index_t w = 0; w < n_layers; w++)
	{
		deltas_w[w] = layers[w].weights * 0;
	}

	while(n_iter--)
	{
		for(index_t c = 0; c < inputs.n_cols; c++)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);	

			vec dl_da = o_layer() - output;
			mat da_dx;

			for(index_t i = n_layers - 1; i > 0; i--)
				layers[i].backprogation_momentum(dl_da, da_dx, deltas_w[i], layers[i - 1], alpha);
	
			layers[0].backprogation_momentum(dl_da, da_dx, deltas_w[0], input, alpha);
		}
	}

	delete [] deltas_w;
}

void network::train(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, size_t n_iter)
{
	init(inputs.n_rows, outputs.n_rows, n_neurons);

	percent_t error;
	while(n_iter--)
	{
		for(index_t c = 0; c < inputs.n_cols; c++)
		{
			const vec & input = inputs.col(c);
			const vec & output = outputs.col(c);

			forward(input, output);	
			
			vec dl_da = o_layer() - output;
			mat da_dx;
	
			for(index_t i = n_layers - 1; i > 0; i--)
				layers[i].backprogation(dl_da, da_dx, layers[i - 1]);
	
			layers[0].backprogation(dl_da, da_dx, input);
		}
	}
}

percent_t network::test(const mat & inputs, const mat & outputs)
{
	percent_t error = 0;
	for(index_t c = 0; c < inputs.n_cols; c++)
	{
		forward(inputs.col(c), outputs.col(c));
		error += loss > 0.01;
	}

	return error / inputs.n_cols;
}
	
void network::init(const size_t & size_in, const size_t & size_out, const vector<size_t> & n_neurons)
{	
	assert(n_neurons.size() == n_layers - 1);

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

