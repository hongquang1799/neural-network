#include "neural-network/optimizer/Optimizer.h"

void mc::SGD::Optimize(Matrix& dE2W, Matrix& W, Matrix& dE2B, Matrix& B)
{
	for (int i = 0; i < W.n_row; i++)
	{
		for (int j = 0; j < W.n_col; j++)
		{
			W(i, j) -= dE2W(i, j) * learning_rate;
		}

		B(i) -= dE2B(i) * learning_rate;
	}
}
