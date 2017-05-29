#include "layer.h"

double sigmoid(const double & x)
{
	return 1 / (1 + exp(-x));  
}

double d_sigmoid(const double & x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

layer::operator const vec & () const
{
	return neurons;
}

void layer::init(const size_t & m, const size_t & n)
{
	weights.resize(m, n);
	weights.randu();
	weights *= 2 * epsilon;
	weights -= epsilon;
}

void layer::forward(const vec & input, function_t f)
{
	neurons = weights.t() * input + 1;
	d_neurons = neurons;
	neurons.transform(f);
}

void layer::backprogation(vec & dl_da, mat & da_dx, const vec & x, function_t df)
{
	d_neurons.transform(df);
	
	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;
		
	dl_da %= d_neurons;

	weights -= eta * (x * dl_da.t());
	
	da_dx = weights * dl_da;
}

