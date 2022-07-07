#include "layer.h"


function_t layer::f = sigmoid;
function_t layer::df = d_sigmoid;

layer::operator const vec & () const
{
	return neurons;
}

void layer::init(const size_t & m, const size_t & n)
{
	weights.randu(m, n);
	weights *= 2 * epsilon;
	weights -= epsilon;
	bias.ones(n);
}

void layer::forward(const vec & input)
{
	neurons = weights.t() * input + bias;
	d_neurons = neurons;
	neurons.transform(f);
}

void layer::backward(vec & dl_da, mat & da_dx, mat & dl_dw, const vec & x)
{
	d_neurons.transform(df);

	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;

	dl_da %= d_neurons;
	da_dx = weights * dl_da;
	dl_dw = x * dl_da.t();
}


double sigmoid(const double & x)
{
	return 1 / (1 + exp(-x));
}

double d_sigmoid(const double & x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

double relu(const double & x)
{
	return x > 0 ? x : 0;
}

double d_relu(const double & x)
{
	return x > 0 ? 1 : 0;
}

