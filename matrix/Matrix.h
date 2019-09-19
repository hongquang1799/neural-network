#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <iostream>
#include <memory>
#include <vector>
#include <Windows.h>

namespace mc
{
	class Matrix
	{
	public:
		Matrix();

		Matrix(size_t row, size_t col);

		~Matrix();

		size_t n_row;
		size_t n_col;

		float * get() const;

		Matrix& allocateMemory(size_t row, size_t col);

		void set(float value);

		void set(const float * arr);

		void set(const std::initializer_list<std::initializer_list<float>>& list);

		void randomize(float min, float max);

		float& operator()(size_t r, size_t c) const;

		float& operator()(size_t i) const;

		//float *operator[](size_t r);

		void log();
	private:
		std::shared_ptr<float> data;
	};
}

void PrintToOutput(const char* szFormat, ...);

#define DEBUG_LOG PrintToOutput

#endif