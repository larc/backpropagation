#include "layer.h"


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

void layer::compute_gradients(vec & dl_da, mat & da_dx, mat & dl_dw, const vec & x)
{
	d_neurons.transform(df);

	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;

	dl_da %= d_neurons;
	da_dx = weights * dl_da;
	dl_dw = x * dl_da.t();
}

void layer::backward_sgd(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x)
{
	d_neurons.transform(df);

	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;

	dl_da %= d_neurons;

	delta_w += x * dl_da.t(); //sgd

	da_dx = weights * dl_da;

	bias -= eta * dl_da;
}

void layer::backward_momentum(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x, const double & alpha)
{
	d_neurons.transform(df);

	if(da_dx.n_rows) //false in output layer to last hidden layer
		dl_da = da_dx;

	dl_da %= d_neurons;

	delta_w = (1 - alpha) * (- eta) * (x * dl_da.t()) + alpha * delta_w;

	da_dx = weights * dl_da;

	weights += delta_w;
	bias -= eta * dl_da;
}

void layer::backward(vec & dl_da, mat & da_dx, const vec & x)
{
	mat dl_dw;
	compute_gradients(dl_da, da_dx, dl_dw, x);

	weights -= eta * dl_dw;
	bias -= eta * dl_da;
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

