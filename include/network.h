#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef double percent_t;

class network
{
	private:
		layer * layers;
		size_t n_layers;
		percent_t loss;
	
	public:
		network(const size_t & h_layers = 0);
		virtual ~network();
		const vec & o_layer() const;
		void forward(const vec & input, const vec & output);
		void backprogation(const vec & input, const vec & output);
		void train(const mat & inputs, const mat & outputs, const size_t & n_neurons, size_t n_iter);
		percent_t test(const mat & inputs, const mat & outputs);
};

#endif // NETWORK_H

