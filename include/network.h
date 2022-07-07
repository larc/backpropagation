#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

#include <vector>


#define TEST

#ifndef TEST
	#define _os_open
	#define _os_error(i, loss)
	#define _os_close
#else
	#define _os_open ofstream os(std::string(SRC_PATH) + "/tmp/iter_loss");
	#define _os_error(i, loss) os << i << " " << loss << endl;
	#define _os_close os.close();
#endif


using namespace std;

class network
{
	public:
		double tol_error = 0.01;	// tolerance error
		double eta = 0.1;			// learning rate

	private:
		vector<layer> layers;
		vector<mat> deltas_w;
		vector<vec> deltas_b;
		double loss;

	public:
		network(const size_t & h_layers = 0);
		const vec & o_layer() const;
		size_t train(const mat & inputs, const mat & outputs, const vector<size_t> & n_neurons, const size_t & n_iter, const size_t & bach_size, const double & alpha);
		double test(const mat & inputs, const mat & outputs);

	private:
		void init(const size_t & size_in, const size_t & size_out, const vector<size_t> & n_neurons);
		void forward(const vec & input, const vec & output);
};


#endif // NETWORK_H

