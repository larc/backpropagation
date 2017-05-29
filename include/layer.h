#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <cmath>

#define eta 1 // learning rate
#define epsilon .00001

using namespace arma;

typedef uword index_t;
typedef double (*function_t) (const double &);

double sigmoid(const double & x);
double d_sigmoid(const double & x);

class layer
{
	public:
		vec neurons;
		vec d_neurons;
		mat weights;

		layer() = default;
		~layer() = default;
		operator const vec & () const;
		void init(const size_t & m, const size_t & n);	
		void forward(const vec & input, function_t f = sigmoid);
		void backprogation(vec & dl_da, mat & da_dx, const vec & x, function_t df = d_sigmoid);
};

#endif // LAYER_H

