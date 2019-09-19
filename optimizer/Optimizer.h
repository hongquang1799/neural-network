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

			std::string name;
		};

		class SGD : public Optimizer
		{
		public:
			SGD() { name = "SGD"; }
			virtual ~SGD() {}

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);
		};

		class Momentum : public Optimizer
		{
		public:
			Momentum() { name = "Momentum"; }
			virtual ~Momentum() {}

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);;
		};

		class Adam : public Optimizer
		{
		public:
			Adam() { name = "Adam"; }
			virtual ~Adam() {}

			virtual void Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B);
		};
	}
}

#endif