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

void network::forward(const vec & input, const vec & output)
{
	layers[0].forward(input);
	for(index_t i = 1; i < n_layers; i++)
		layers[i].forward(layers[i-1]);
	
	loss = norm(output - o_layer());
	loss = 0.5 * loss * loss;
}

void network::backprogation(const vec & input, const vec & output)
{
	vec dl_da = o_layer() - output;
	mat da_dx;
	
	for(index_t i = n_layers - 1; i > 0; i--)
		layers[i].backprogation(dl_da, da_dx, layers[i - 1]);
	
	layers[0].backprogation(dl_da, da_dx, input);
}

void network::train(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, size_t n_iter)
{
	assert(n_neurons.size() == n_layers - 1);

	layers[0].init(inputs.n_rows, n_neurons.front());
	for(index_t i = 1; i < n_neurons.size(); i++)
		layers[i].init(n_neurons[i - 1], n_neurons[i]);
	layers[n_layers - 1].init(n_neurons.back(), outputs.n_rows);

	percent_t error;
	while(n_iter--)
	{
		for(index_t c = 0; c < inputs.n_cols; c++)
		{
			forward(inputs.col(c), outputs.col(c));	
			backprogation(inputs.col(c), outputs.col(c));
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

