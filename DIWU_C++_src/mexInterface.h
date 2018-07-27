#pragma once

#include "mex.h"
#include "Mat.h"
#include "../mex_opencv_2.4/MxArray.hpp"
#include "../mex_opencv_2.4/MexAsserts.h"

namespace diwu
{
	struct Idx
	{
		// outputs
		static const int OUT_Diversity_score_Map = 0;

		// inputs
		static const int IN_DIVERSITY_TYPE = 0;
		static const int IN_NNF = 1;
		static const int IN_windowSizeM = 2;
		static const int IN_windowSizeN = 3;

		//version 1
		static const int IN_differentWeightStartIndex = 4;
		static const int IN_differentWeight = 5;

		//version 2
		static const int IN_weightsVec = 4;

		//version 3
		static const int IN_xyPositons = 4;
		static const int IN_h = 5;
		static const int IN_INV_FREQ = 6;
		static const int IN_NNF_TRANSPOSE = 7;
		static const int IN_INV_FREQ_TRANSPOSE = 8;
	};




	inline Mat<int> ExtractNnfAndWindow(const mxArray** inputs, int& windowM, int& windowN)
	{
		const mxArray* pNnfMx = inputs[Idx::IN_NNF];
		int* nnfArray = (int32_T *)mxGetData(pNnfMx);
		Mat<int> nnfMat = Mat<int>(nnfArray, static_cast<int>(mxGetN(pNnfMx)), static_cast<int>(mxGetM(pNnfMx)));

		/*  loading window parameterss */
		windowM = MxArray(inputs[Idx::IN_windowSizeM]).toInt();
		windowN = MxArray(inputs[Idx::IN_windowSizeN]).toInt();
		return nnfMat;
	}

	inline Mat<int> ExtractNnfTransposeAndWindow(const mxArray** inputs)
	{
		const mxArray* pNnfMx = inputs[Idx::IN_NNF];
		int* nnfArray = (int32_T *)mxGetData(pNnfMx);
		Mat<int> nnfMat = Mat<int>(nnfArray, static_cast<int>(mxGetN(pNnfMx)), static_cast<int>(mxGetM(pNnfMx)));

		return nnfMat;
	}

	inline Mat<float> ExtractInvFreq(const mxArray** inputs)
	{
		const mxArray* pInvFreqMx = inputs[Idx::IN_INV_FREQ];
		float* invFreqArray = (REAL32_T *)mxGetData(pInvFreqMx);
		Mat<float> invFreqMat = Mat<float>(invFreqArray, static_cast<int>(mxGetN(pInvFreqMx)), static_cast<int>(mxGetM(pInvFreqMx)));

		return invFreqMat;
	}

	inline Mat<double> ExtractInvFreqTranspose(const mxArray** inputs)
	{
		const mxArray* pInvFreqMx = inputs[Idx::IN_INV_FREQ_TRANSPOSE];
		double* invFreqArray = (REAL64_T *)mxGetData(pInvFreqMx);
		Mat<double> invFreqMat = Mat<double>(invFreqArray, static_cast<int>(mxGetN(pInvFreqMx)), static_cast<int>(mxGetM(pInvFreqMx)));

		return invFreqMat;
	}

	inline Mat<float> CreateOutputScoreMap(mxArray** outputs, int windowM, int windowN, int nnfM, int nnfN)
	{
		int outputN = nnfN - windowN + 1;
		int outputM = nnfM - windowM + 1;
		ASSERT((nnfN - windowN + 1) > 0);
		ASSERT((nnfM - windowM + 1) > 0);
		size_t dimensions[2] = { outputM, outputN };
		// create a mxArray of int32 of size == dimensions
		mxArray* pDiversityScoreMap_mxArray = mxCreateNumericArray(2, dimensions, mxSINGLE_CLASS, mxREAL);
		outputs[Idx::OUT_Diversity_score_Map] = pDiversityScoreMap_mxArray;
		Mat<float> diversityScoreMap = Mat<float>((float *)mxGetData(pDiversityScoreMap_mxArray), outputN, outputM);
		return diversityScoreMap;
	}
}
