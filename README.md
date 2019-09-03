# simple-neural-network
 simple c++ library for machine learning
 
 example:
 
#include "neural-network/matrix/Matrix.h"
#include "neural-network/FNN.h"
#include "neural-network/layer/Dense.h"
#include "time.h"

using namespace mc;

void Example()
{
	int n_row = 1;
	int n_col = 2;

	float inputs[] = { 0.05f, 0.1f };
	Matrix input(n_row, n_col);
	input.set(inputs);

	float weights_1[] = { 0.15f, 0.2f, 0.25f, 0.3f };
	float biases_1[] = { 0.35f, 0.35f };

	float weights_2[] = { 0.4f, 0.45f, 0.5f, 0.55f };
	float biases_2[] = { 0.6f, 0.6f };

	FNN nn;
	nn.SetInputUnit(n_col);

	nn.AddDense(2, "Sigmoid", weights_1, biases_1);
	nn.AddDense(2, "Sigmoid", weights_2, biases_2);

	nn.SetLoss("MeanSquarredError");
	nn.SetOptimizer("SGD", 0.5f);
	nn.Log();

	Matrix output = nn.Predict(input);

	float targets[] = { 0.01f, 0.99f };
	Matrix target(1, 2);
	target.set(targets);

	nn.Train(input, target);
	
	DEBUG_LOG("Output:\n");
	output.log();
}

void main()
{
	srand(time(NULL));

	Example();

	getchar();
}
