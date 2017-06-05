#include "layer.h"

double sigmoid(const double & x)
{
	return 1 / (1 + exp(-x));  
}

double d_sigmoid(const double & x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

function_t layer::f = sigmoid;
function_t layer::df = d_sigmoid;

void layer::set_activation(function_t _f, function_t _df)
{
	f = _f;
	df = _df;
}

layer::operator const vec & () const
{
	return neurons;
}

void layer::init(const size_t & m, const size_t & n)
{
	weights.resize(m, n);
	weights.randu();

	bias.resize(n);
	bias.ones();
}

void layer::forward(const vec & input)
{
	neurons = weights.t() * input + bias;
	d_neurons = neurons;
	neurons.transform(f);
}

void layer::backprogation_momentum(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x, const percent_t & alpha)
{
	d_neurons.transform(df);
	
	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;
		
	dl_da %= d_neurons;

	delta_w = (1 - alpha) * (- eta) * (x * dl_da.t()) + alpha * delta_w;
	weights += delta_w;
	bias -= eta * dl_da;
	
	da_dx = weights * dl_da;
}

void layer::backprogation(vec & dl_da, mat & da_dx, const vec & x)
{
	d_neurons.transform(df);
	
	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;
		
	dl_da %= d_neurons;

	weights -= eta * (x * dl_da.t());
	bias -= eta * dl_da;
	
	da_dx = weights * dl_da;
}

