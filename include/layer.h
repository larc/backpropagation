#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <armadillo>

#define epsilon .001

using namespace arma;

using uint_t = unsigned int;
using function_t = double (*) (const double &);

class layer
{
	public:
		static function_t f;
		static function_t df;

	public:
		vec neurons;
		vec d_neurons;
		mat weights;
		vec bias;

		operator const vec & () const;
		void init(const size_t & m, const size_t & n);
		void forward(const vec & input);
		void backward(vec & dl_da, mat & da_dx, mat & dl_dw, const vec & x);
};


// activation functions and its derivatives

double sigmoid(const double & x);
double d_sigmoid(const double & x);

double relu(const double & x);
double d_relu(const double & x);


#endif // LAYER_H

