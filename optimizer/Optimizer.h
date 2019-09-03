#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neural-network/layer/Layer.h"
#include "neural-network/matrix/Matrix.h"

namespace mc
{
	namespace opt
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

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);
		};

		class Momentum : public Optimizer
		{
		public:
			Momentum() {}
			virtual ~Momentum() {}

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);;
		};

		class Adam : public Optimizer
		{
		public:
			Adam() {}
			virtual ~Adam() {}

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);
		};
	}
}

#endif