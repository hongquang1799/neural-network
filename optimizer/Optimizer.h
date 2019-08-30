#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neural-network/layer/Layer.h"
#include "neural-network/matrix/Matrix.h"

namespace mc
{
	class Optimizer
	{
	public:
		Optimizer() {}
		virtual ~Optimizer() {}

		virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B) = 0;

		void Log() {}

		float learning_rate;
	};

	class SGD : public Optimizer
	{
	public:
		SGD() {}
		virtual ~SGD() {}

		void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);
	};
}

#endif