#ifndef LAYER_H
#define LAYER_H

#include <armadillo>

using namespace arma;

class layer
{
	private:
		vec neurons;
		mat weights;
	
	public:
		layer();
		~layer();
		void forward(const vec & inputs);
};



#endif // LAYER_H

