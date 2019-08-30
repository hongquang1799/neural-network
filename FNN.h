#ifndef FNN_H
#define FNN_H

#include "vector"
#include "string"
#include "neural-network/matrix/Matrix.h"

namespace mc
{
	class FNN // feed-forward neural network
	{
	public:
		FNN();
		virtual ~FNN();

		void SetInputUnit(size_t unit);
		class Dense * AddDense(size_t unit, const std::string& activation);
		class Dense * AddDense(size_t unit, const std::string& activation, const std::vector<std::vector<float>>& W, const std::vector<std::vector<float>>& B);
		void SetLoss(const std::string& loss);
		void SetOptimizer(const std::string& optimize, float learning_rate);

		Matrix Predict(Matrix& input);
		void Train(Matrix& input, Matrix& target);

		void Log();
	private:
		size_t inputUnit;

		std::vector<class Dense*> listOfDenses;

		std::shared_ptr<class LossFunc> lossFunc;

		std::shared_ptr<class Optimizer> optimizer;
	};
}

#endif
