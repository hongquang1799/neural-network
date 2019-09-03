#include "neural-network/FNN.h"
#include "neural-network/layer/Dense.h"
#include "neural-network/layer/AtvtFunc.h"
#include "neural-network/loss/LossFunc.h"
#include "neural-network/optimizer/Optimizer.h"

using namespace mc;
using namespace mc::opt;
using namespace mc::loss;

mc::FNN::FNN()
{
}

mc::FNN::~FNN()
{
	for (auto dense : listOfDenses)
	{
		delete dense;
	}
}

Matrix mc::FNN::Predict(Matrix& input)
{
	Matrix A_in = input;
	for (auto dense : listOfDenses)
	{
		A_in = dense->Forward(A_in);
	}

	return A_in;
}

void mc::FNN::Train(Matrix& input, Matrix& target)
{
	Matrix predict = Predict(input);
	Matrix dE = lossFunc->Gradient(predict, target);
	DEBUG_LOG("dError:\n");
	dE.log();
	

	for (auto it = this->listOfDenses.rbegin(); it != this->listOfDenses.rend(); it++)
	{
		Dense * dense = (*it);
		dE = dense->Backward(dE);

		optimizer->Optimize(dense->dE2W, dense->W, dense->dE2S, dense->B);

		DEBUG_LOG("W:\n");
		dense->W.log();
	}
}

void mc::FNN::SetInputUnit(size_t unit)
{
	inputUnit = unit;
}

Dense * mc::FNN::AddDense(size_t unit, const std::string& activation)
{
	Dense * dense = NULL;

	if (listOfDenses.size() == 0)
	{
		dense = new Dense(unit, inputUnit);	
		listOfDenses.push_back(dense);
	}
	else
	{
		dense = new Dense(unit, listOfDenses.back()->n_row);
		listOfDenses.push_back(dense);
	}
	
	if (activation == "Sigmoid")
		dense->atvt = new Sigmoid();
	else if (activation == "Relu")
		dense->atvt = new ReLU();
	else if (activation == "Softmax")
		dense->atvt = new Softmax();

	dense->atvt->name = activation;

	return dense;
}

class Dense * mc::FNN::AddDense(size_t unit, const std::string& activation, const float weights[], const float biases[])
{
	Dense * dense = AddDense(unit, activation);
	dense->W.set(weights);
	dense->B.set(biases);

	return dense;
}

void mc::FNN::SetLoss(const std::string& loss)
{
	if (loss == "MeanSquarredError")
	{
		lossFunc = std::shared_ptr<LossFunc>(new MeanSquarredError());
	}
}

void mc::FNN::SetOptimizer(const std::string& optimize, float learning_rate)
{
	if (optimize == "SGD")
	{
		optimizer = std::shared_ptr<Optimizer>(new SGD());
	}
	else if (optimize == "Momentum")
	{
		optimizer = std::shared_ptr<Momentum>(new Momentum());
	}
	else if (optimize == "Adam")
	{
		optimizer = std::shared_ptr<Adam>(new Adam());
	}

	optimizer->learning_rate = learning_rate;
}

void mc::FNN::Log()
{
	DEBUG_LOG("Input Unit: %d\n", inputUnit);
	for (auto dense : listOfDenses)
	{
		dense->Log();
	}
}


