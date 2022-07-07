#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <cmath>

#define eta .1 // learning rate
#define epsilon .001

using namespace arma;

using uint_t = unsigned int;
using function_t = double (*) (const double &);

class layer
{
	private:
		static function_t f;
		static function_t df;

	public:
		static void set_activation(function_t _f, function_t _df);

	public:
		vec neurons;
		vec d_neurons;
		mat weights;
		vec bias;

		layer() = default;
		~layer() = default;
		operator const vec & () const;
		void init(const size_t & m, const size_t & n);
		void forward(const vec & input);
		void compute_gradients(vec & dl_da, mat & da_dx, mat & dl_dw, const vec & x);
		void backward_sgd(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x);
		void backward_momentum(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x, const double & alpha);
		void backward(vec & dl_da, mat & da_dx, const vec & x);
};


// activation functions and its derivatives

double sigmoid(const double & x);
double d_sigmoid(const double & x);

double relu(const double & x);
double d_relu(const double & x);


#endif // LAYER_H

