#include "layer.h"

layer::layer()
{
}

layer::~layer()
{
}

void forward(const vec & inputs)
{
	neurons = weights * inputs;
}


