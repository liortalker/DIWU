#pragma once

#include "mex.h"

namespace diwu
{
	void IWU(mxArray** outputs, int inputsElementCount, const mxArray** inputs, unsigned nthreads = 0);
	void DIWU(mxArray** outputs, int inputsElementCount, const mxArray** inputs, unsigned nthreads = 0);
	void saveIm(cv::Mat image);

}