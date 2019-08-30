#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

#include "neural-network/loss/LossFunc.h"
#include "neural-network/matrix/Matrix.h"

namespace mc
{
	class LossFunc 
	{
	public:
		LossFunc();
		virtual ~LossFunc();

		virtual Matrix Get(Matrix& predict, Matrix& target) = 0;
		virtual Matrix Gradient(Matrix& predict, Matrix& target) = 0;
	private:
	};

	class MeanSquarredError : public LossFunc
	{
	public:
		MeanSquarredError() {}
		virtual ~MeanSquarredError() {}

		Matrix Get(Matrix& predict, Matrix& target) override;
		Matrix Gradient(Matrix& predict, Matrix& target) override;
	};
}

#endif