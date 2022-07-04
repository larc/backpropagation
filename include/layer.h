#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <cmath>

#define eta .1 // learning rate
#define epsilon .001

using namespace arma;


typedef uword index_t;
typedef double percent_t;
typedef double (*function_t) (const double &);

double sigmoid(const double & x);
double d_sigmoid(const double & x);

class network;

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
		void backpropagation_sgd(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x);
		void backpropagation_momentum(vec & dl_da, mat & da_dx, mat & delta_w, const vec & x, const percent_t & alpha);
		void backpropagation(vec & dl_da, mat & da_dx, const vec & x);

	friend network;
};


#endif // LAYER_H

