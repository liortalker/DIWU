#include <algorithm>
#include "mexInterface.h"
#include "../mex_opencv_2.4/MxArray.hpp"
#include "../mex_opencv_2.4/EigenExtensions.h"
#include "../mex_opencv_2.4/MexAsserts.h"
#include "CountArray.h"
#include "tictoc.h"
#include <math.h> 
#include <thread>
#include "DIWU.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace diwu;


inline Mat<int> CreateZeroMatInt(int nnfM, int nnfN)
{
	size_t dimensions[2] = { nnfM, nnfN };
	// create a mxArray of int32 of size == dimensions
	mxArray* pDiversityScoreMap_mxArray = mxCreateNumericArray(2, dimensions, mxSINGLE_CLASS, mxREAL);
	Mat<int> diversityScoreMap = Mat<int>((int *)mxGetData(pDiversityScoreMap_mxArray), nnfN, nnfM);
	return diversityScoreMap;
}

inline Mat<long> CreateZeroMatLong(int nnfM, int nnfN)
{
	size_t dimensions[2] = { nnfM, nnfN };
	// create a mxArray of int32 of size == dimensions
	mxArray* pDiversityScoreMap_mxArray = mxCreateNumericArray(2, dimensions, mxSINGLE_CLASS, mxREAL);
	Mat<long> diversityScoreMap = Mat<long>((long *)mxGetData(pDiversityScoreMap_mxArray), nnfN, nnfM);
	return diversityScoreMap;
}

inline Mat<double> CreateZeroMatDouble(int nnfM, int nnfN)
{
	size_t dimensions[2] = { nnfM, nnfN };
	// create a mxArray of int32 of size == dimensions
	mxArray* pDiversityScoreMap_mxArray = mxCreateNumericArray(2, dimensions, mxSINGLE_CLASS, mxREAL);
	Mat<double> diversityScoreMap = Mat<double>((double *)mxGetData(pDiversityScoreMap_mxArray), nnfN, nnfM);
	return diversityScoreMap;
}

inline Mat<float> CreateZeroMatFloat(int nnfM, int nnfN)
{
	size_t dimensions[2] = { nnfM, nnfN };
	// create a mxArray of int32 of size == dimensions
	mxArray* pDiversityScoreMap_mxArray = mxCreateNumericArray(2, dimensions, mxSINGLE_CLASS, mxREAL);
	Mat<float> diversityScoreMap = Mat<float>((float *)mxGetData(pDiversityScoreMap_mxArray), nnfN, nnfM);
	return diversityScoreMap;
}

inline cv::Mat toCVMatFloat(Mat<float> inMat) {
	cv::Mat mat(inMat.height, inMat.width, CV_32F);
	for (int i = 0; i < inMat.height; i++) {
		for (int j = 0; j < inMat.width; j++) {
			mat.at<float>(i,j) = inMat(i,j);
		}
	}
	return mat;
}

enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,

	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};

int FindMaxNNIndex(const Mat<int>& nnfMat)
{
	int nnfArrayLength = nnfMat.width * nnfMat.height;
	const int* nnfArray = &nnfMat(0, 0);
	int maxElement = *std::max_element(nnfArray, nnfArray + nnfArrayLength);
	return maxElement;
}

struct iwuScanParams
{
	int startColIndexInclusive;
	int endColIndexExclusive;
	const Mat<int>* pNnfMat;
	const Mat<float>* invFreqMat;
	Mat<float>* pScoreMap;
	const Map<MatrixXi>* pXyPositions;
	int windowM;
	int windowN;
	double h;
	int maxElement;

	// default empty Ctor for array constraction
	iwuScanParams(){}

	iwuScanParams(
		int startColIndexInclusive,
		int endColIndexExclusive,
		int windowM, int windowN, int h,
		const Map<MatrixXi>* pXyPositions, Mat<float>* pScoreMap, const Mat<int>* pNnfMat, const Mat<float>* pInvFreqMat, int maxElement) :
		startColIndexInclusive(startColIndexInclusive), endColIndexExclusive(endColIndexExclusive),
		pNnfMat(pNnfMat), invFreqMat(pInvFreqMat), pScoreMap(pScoreMap), pXyPositions(pXyPositions),
		windowM(windowM), windowN(windowN), h(h), maxElement(maxElement)
	{
	}
};

struct diwuScanParams
{
	const Mat<int>* pNnfMat;
	const Mat<float>* invFreqMat;
	Mat<float>* pScoreMapX;
	Mat<float>* pScoreMapY;
	const Map<MatrixXi>* pXyPositions;
	int windowM;
	int windowN;

	// default empty Ctor for array constraction
	diwuScanParams(){}

	diwuScanParams(
		int windowM, int windowN, const Map<MatrixXi>* pXyPositions, Mat<float>* pScoreMapX, Mat<float>* pScoreMapY, const Mat<int>* pNnfMat, const Mat<float>* pInvFreqMat) :
		pNnfMat(pNnfMat), invFreqMat(pInvFreqMat), pScoreMapX(pScoreMapX), pScoreMapY(pScoreMapY), pXyPositions(pXyPositions), windowM(windowM), windowN(windowN)	{	}
};

void OneDimL1Dist(long& dOld, long& drOld, Mat<int>& oldHist, int& zeroPointer, int newDelta, int oldDelta, int w, int histSize, int rowIdx) {

	dOld = dOld + drOld + abs(newDelta) - abs(oldDelta);
	int oldIdx = (oldDelta + zeroPointer + histSize) % (histSize);
	oldHist(rowIdx, oldIdx) = oldHist(rowIdx, oldIdx) - 1;
	int returnValOld;
	if ((oldIdx - zeroPointer + histSize) % (histSize) <= w) {
		returnValOld = 1;
	}
	else {
		returnValOld = -1;
	}
	zeroPointer = (zeroPointer - 1 + histSize) % (histSize);
	int newIdx = (newDelta + zeroPointer + histSize) % (histSize);
	int returnValNew;
	if ((newIdx - zeroPointer + histSize) % (histSize) <= w) {
		returnValNew = 1;
	}
	else {
		returnValNew = -1;
	}
	drOld = drOld + 2 * oldHist(rowIdx, zeroPointer) - returnValOld + returnValNew;
	oldHist(rowIdx, newIdx) = oldHist(rowIdx, newIdx) + 1;
}

void IWURasterScan(Mat<int> nnfMat, Mat<float> invFreqMat, int windowM, int windowN, const Map<MatrixXi>& xyPositions, double h,
	Mat<float>& scoreMap_out, int startColIndexInclusive, int endColIndexExclusive, int* pMaxElement = nullptr)
{
	int maxElement;
	if (pMaxElement == nullptr)
	{
		maxElement = FindMaxNNIndex(nnfMat);
	}
	else
	{
		maxElement = *pMaxElement;
	}

	ASSERT2(maxElement <= xyPositions.cols(), "the amount of xy positions must be greater or equal to the maximum index in the nnf matrix");

	int rowCount = nnfMat.height;
	int colCount = nnfMat.width;
	DeformableDiversityWeightArray windowItemsWeights(1 + maxElement, h);

	int histSize = 2*windowN + 1;
	Mat<long> ourSumVec = CreateZeroMatLong(rowCount, nnfMat.width);
	Mat<int> histMat1 = CreateZeroMatInt(rowCount, histSize);
	Mat<int> histMat2 = CreateZeroMatInt(rowCount, histSize);
	Mat<int> drVec1 = CreateZeroMatInt(rowCount, 1);
	Mat<int> drVec2 = CreateZeroMatInt(rowCount, 1);
	Mat<long> dSumVec1 = CreateZeroMatLong(rowCount, 1);
	Mat<long> dSumVec2 = CreateZeroMatLong(rowCount, 1);
	Mat<long> tempSumVec = CreateZeroMatLong(colCount, 1);
	Mat<int> zeroPointerVec1 = CreateZeroMatInt(rowCount, 1);
	Mat<int> zeroPointerVec2 = CreateZeroMatInt(rowCount, 1);
	int winSum;
	int rowSum;
	int curDiff;
	int idx;


	// go over the left most windows and calculate them as preprocessing for the whole image
	for (int k = 0; k < rowCount; k++) {
		//for (int i = 0; i < windowM; i++) {
		rowSum = 0;
		for (int j = 0; j < windowN; j++) {
			curDiff = xyPositions(0,nnfMat(k, j)) - j;
			rowSum = rowSum + abs(curDiff);
			idx = curDiff + windowN;
			histMat1(k, idx) = histMat1(k, idx) + 1;
			histMat2(k, idx) = histMat1(k, idx);
		}
		for (int j = 0; j < windowN; j++) {
			drVec1(k, 0) = drVec1(k, 0) - histMat1(k, j);
		}
		for (int j = windowN; j < histSize; j++) {
			drVec1(k, 0) = drVec1(k, 0) + histMat1(k, j);
		}
		drVec1(k, 0) = drVec1(k, 0) - 1;
		drVec2(k, 0) = drVec1(k, 0);
		dSumVec1(k, 0) = rowSum;
		dSumVec2(k, 0) = rowSum;
		zeroPointerVec1(k, 0) = windowN;
		zeroPointerVec2(k, 0) = windowN;
	}

	// prepare the sum of windows
	winSum = 0;
	for (int k = 0; k < rowCount; k++) {
		if (k < windowM) {
			winSum = winSum + dSumVec1(k, 0);
		}
		else {
			ourSumVec(k - windowM, 0) = winSum;
			winSum = winSum - dSumVec1(k-windowM, 0) + dSumVec1(k, 0);
		}
	}

	
	for (int k = 0; k < colCount - windowN; k++) {
		winSum = 0;
		for (int i = 0; i < windowN; i++) {
			for (int j = 0; j < windowM; j++) {
				curDiff = xyPositions(0, nnfMat(j, k+i)) - i;
				winSum = winSum + abs(curDiff);
			}
		}
		ourSumVec(0, k) = winSum;
	}


	int curRow;
	int curCol;
	int delta1, delta2, delta3;
	int e11, e22, e12, e21;
	long oldDr, dOld;
	int newDelta, oldDelta, zeroPointer;

	for (int i = 1; i < rowCount - windowM; i++) {

		for (int k = 1; k < colCount - windowN; k++) {
			curRow = i - 1;
			curCol = k - 1;
			delta1 = ourSumVec(curRow, curCol + 1);
			delta2 = ourSumVec(curRow, curCol);
			delta3 = ourSumVec(curRow + 1, curCol);
			e11 = xyPositions(0, nnfMat(curRow, curCol)) - 0;
			e22 = windowN -1 - xyPositions(0, nnfMat(curRow + windowM, curCol + windowN));
			e12 = windowN -1 - xyPositions(0, nnfMat(curRow, curCol + windowN));
			e21 = xyPositions(0, nnfMat(curRow + windowM, curCol)) - 0;
			// calculate the sum using the 3 up, left and up-left neighbours
			ourSumVec(curRow + 1, curCol + 1) = delta1 + delta3 - delta2 + drVec2(curRow + windowM, 0) - drVec1(curRow, 0) + e11 + e22 - e12 - e21;
			scoreMap_out(curRow + 1, curCol + 1) = ourSumVec(curRow + 1, curCol + 1);

			// r1
			oldDr = drVec1(curRow, 0);
			dOld = dSumVec1(curRow, 0);
			newDelta = xyPositions(0, nnfMat(curRow, k + windowN - 1)) - (windowN-1);
			oldDelta = xyPositions(0, nnfMat(curRow, k - 1)) - 0;
			zeroPointer = zeroPointerVec1(curRow, 0);
			OneDimL1Dist(dOld, oldDr, histMat1, zeroPointer, newDelta, oldDelta, windowN, histSize, curRow);
			drVec1(curRow, 0) = oldDr;
			dSumVec1(curRow, 0) = dOld;
			zeroPointerVec1(curRow, 0) = zeroPointer;

			// r2
			oldDr = drVec2(curRow + windowM, 0);
			dOld = dSumVec2(curRow + windowM, 0);
			newDelta = xyPositions(0, nnfMat(curRow + windowM, k + windowN - 1)) - (windowN-1);
			oldDelta = xyPositions(0, nnfMat(curRow + windowM, k - 1)) - 0;
			zeroPointer = zeroPointerVec2(curRow, 0);
			OneDimL1Dist(dOld, oldDr, histMat2, zeroPointer, newDelta, oldDelta, windowN, histSize, curRow + windowM);
			drVec2(curRow + windowM, 0) = oldDr;
			dSumVec2(curRow + windowM, 0) = dOld;
			zeroPointerVec2(curRow, 0) = zeroPointer;

		}
	}

}

inline float exp1(float x) {
	x = 1.0 + x / 256.0;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

void DIWURasterScanX(Mat<int> nnfMat, Mat<float> invFreqMat, int windowM, int windowN, const Map<MatrixXi>& xyPositions, Mat<float>& scoreMap_out, bool isX)
{

	int rowCount = nnfMat.height;
	int colCount = nnfMat.width;

	int histSizeX = 2 * windowN + 1;
	Mat<int> zeroPointerVec1PosX = CreateZeroMatInt(rowCount, 1);
	Mat<int> zeroPointerVec2PosX = CreateZeroMatInt(rowCount, 1);
	Mat<int> zeroPointerVec1NegX = CreateZeroMatInt(rowCount, 1);
	Mat<int> zeroPointerVec2NegX = CreateZeroMatInt(rowCount, 1);
	Mat<float> ourSumVecPosX = CreateZeroMatFloat(rowCount, colCount);
	Mat<float> ourSumVecNegX = CreateZeroMatFloat(rowCount, colCount);
	Mat<float> histMat1PosX = CreateZeroMatFloat(rowCount, histSizeX);
	Mat<float> histMat2PosX = CreateZeroMatFloat(rowCount, histSizeX);
	Mat<float> histMat1NegX = CreateZeroMatFloat(rowCount, histSizeX);
	Mat<float> histMat2NegX = CreateZeroMatFloat(rowCount, histSizeX);
	Mat<float> dSumVec1PosX = CreateZeroMatFloat(rowCount, 1);
	Mat<float> dSumVec2PosX = CreateZeroMatFloat(rowCount, 1);
	Mat<float> dSumVec1NegX = CreateZeroMatFloat(rowCount, 1);
	Mat<float> dSumVec2NegX = CreateZeroMatFloat(rowCount, 1);
	Mat<float> tempColSumVecPosX = CreateZeroMatFloat(colCount, 1);
	Mat<float> tempColSumVecNegX = CreateZeroMatFloat(colCount, 1);

	float winSumPosX;
	float winSumNegX;
	int curDiffX;
	float curRowSumDiffNegX;
	float curRowSumDiffPosX;
	int idxX;
	int xyPositionIdx;

	if (isX) {
		xyPositionIdx = 0;
	}
	else {
		xyPositionIdx = 1;
	}

	// LEFT & RIGHT COLUMNS - go over the left and right most windows and calculate them as preprocessing step for the whole image
	winSumPosX = 0;
	winSumNegX = 0;
	int jRev;
	for (int k = 0; k < rowCount; k++) {
		/// X
		curRowSumDiffNegX = 0.0;
		curRowSumDiffPosX = 0.0;
		for (int j = 0; j < windowN; j++) {
			// X
			curDiffX = xyPositions(xyPositionIdx, nnfMat(k, j)) - j;
			if (curDiffX >= 0) {
				curRowSumDiffPosX = curRowSumDiffPosX + invFreqMat(k, j)*exp1(-abs(curDiffX));
			}

			idxX = curDiffX + windowN;
			histMat1PosX(k, idxX) = histMat1PosX(k, idxX) + invFreqMat(k, j);
			histMat2PosX(k, idxX) = histMat1PosX(k, idxX);

			jRev = colCount - windowN + j;
			curDiffX = xyPositions(xyPositionIdx, nnfMat(k, jRev)) - (jRev - (colCount - windowN));
			if (curDiffX < 0) {
				curRowSumDiffNegX = curRowSumDiffNegX + invFreqMat(k, jRev)*exp1(-abs(curDiffX));
			}
			idxX = curDiffX + windowN;
			histMat1NegX(k, idxX) = histMat1NegX(k, idxX) + invFreqMat(k, jRev);
			histMat2NegX(k, idxX) = histMat1NegX(k, idxX);

		}

		dSumVec1PosX(k, 0) = curRowSumDiffPosX;
		dSumVec2PosX(k, 0) = curRowSumDiffPosX;
		zeroPointerVec1PosX(k, 0) = windowN;
		zeroPointerVec2PosX(k, 0) = windowN;
		dSumVec1NegX(k, 0) = curRowSumDiffNegX;
		dSumVec2NegX(k, 0) = curRowSumDiffNegX;
		zeroPointerVec1NegX(k, 0) = windowN;
		zeroPointerVec2NegX(k, 0) = windowN;

		if (k < windowM) {
			winSumPosX = winSumPosX + dSumVec1PosX(k, 0);
			winSumNegX = winSumNegX + dSumVec1NegX(k, 0);
		}
		else {
			ourSumVecPosX(k - windowM, 0) = winSumPosX;
			winSumPosX = winSumPosX - dSumVec1PosX(k - windowM, 0) + dSumVec1PosX(k, 0);
			ourSumVecNegX(k - windowM, colCount - windowN) = winSumNegX;
			winSumNegX = winSumNegX - dSumVec1NegX(k - windowM, 0) + dSumVec1NegX(k, 0);
		}

	}

	ourSumVecPosX(rowCount - windowM, 0) = winSumPosX;
	ourSumVecNegX(rowCount - windowM, colCount - windowN) = winSumNegX;


	// calculate sum of windows in the first rows for X (naively)
	for (int k = 0; k <= colCount - windowN; k++) {
		winSumPosX = 0;
		winSumNegX = 0;
		for (int i = 0; i < windowN; i++) {
			for (int j = 0; j < windowM; j++) {
				curDiffX = xyPositions(xyPositionIdx, nnfMat(j, k + i)) - i;
				if (curDiffX < 0) {
					winSumNegX = winSumNegX + invFreqMat(j, k + i)*exp1(-abs(curDiffX));
				}
				else {
					winSumPosX = winSumPosX + invFreqMat(j, k + i)*exp1(-abs(curDiffX));
				}
			}
		}
		ourSumVecPosX(0, k) = winSumPosX;
		ourSumVecNegX(0, k) = winSumNegX;
	}


	int curRow;
	int curCol;
	float delta1PosX, delta2PosX, delta3PosX;
	float delta1NegX, delta2NegX, delta3NegX;
	float newWeight1X, newWeight2X, oldWeight1X, oldWeight2X;
	float newWeight1NegX, newWeight2NegX, oldWeight1NegX, oldWeight2NegX;
	float e11X, e22X, e12X, e21X, e22NegX, e12NegX;
	int newDelta1X, newDelta2X, oldDelta1X, oldDelta2X;
	int newDelta1NegX, newDelta2NegX, oldDelta1NegX, oldDelta2NegX;
	int w = windowN, w2 = colCount;

	for (int i = 1; i <= rowCount - windowM; i++) {

		for (int k = 1; k <= colCount - w; k++) {
			curRow = i - 1;
			curCol = k - 1;
			// POSITIVE
			delta1PosX = ourSumVecPosX(curRow, curCol + 1);
			delta2PosX = ourSumVecPosX(curRow, curCol);
			delta3PosX = ourSumVecPosX(curRow + 1, curCol);
			oldDelta1X = xyPositions(xyPositionIdx, nnfMat(curRow, curCol)) - 0;
			oldWeight1X = invFreqMat(curRow, curCol);
			e11X = oldWeight1X*exp1(-oldDelta1X);
			newDelta2X = xyPositions(xyPositionIdx, nnfMat(curRow + windowM, curCol + w)) - (w - 1);
			newWeight2X = invFreqMat(curRow + windowM, curCol + w);
			e22X = abs((float)newDelta2X);
			if (newDelta2X == 0) {
				e22X = newWeight2X;
			}
			else {
				e22X = 0;
			}
			newDelta1X = xyPositions(xyPositionIdx, nnfMat(curRow, curCol + w)) - (w - 1);
			newWeight1X = invFreqMat(curRow, curCol + w);
			e12X = abs((float)newDelta1X);
			if (newDelta1X == 0) {
				e12X = newWeight1X;
			}
			else {
				e12X = 0;
			}
			oldDelta2X = xyPositions(xyPositionIdx, nnfMat(curRow + windowM, curCol)) - 0;
			oldWeight2X = invFreqMat(curRow + windowM, curCol);
			e21X = oldWeight2X*exp1(-abs(oldDelta2X));
			zeroPointerVec1PosX(curRow, 0) = (zeroPointerVec1PosX(curRow, 0) - 1 + histSizeX) % (histSizeX);
			zeroPointerVec2PosX(curRow + windowM, 0) = (zeroPointerVec2PosX(curRow + windowM, 0) - 1 + histSizeX) % (histSizeX);
			float firstRowDiffX = (exp1(-1) - 1)*(dSumVec1PosX(curRow, 0) - e11X) + histMat1PosX(curRow, zeroPointerVec1PosX(curRow, 0));
			float lastRowDiffX = (exp1(-1) - 1)*(dSumVec2PosX(curRow + windowM, 0) - e21X) + histMat2PosX(curRow + windowM, zeroPointerVec2PosX(curRow + windowM, 0));
			// calculate the sum using the 3 up, left and up-left neighbours
			ourSumVecPosX(curRow + 1, curCol + 1) = delta1PosX + delta3PosX - delta2PosX + lastRowDiffX - firstRowDiffX + e11X + e22X - e12X - e21X;


			int oldIdx1X = (oldDelta1X + zeroPointerVec1PosX(curRow, 0) + 1 + histSizeX) % histSizeX;
			int oldIdx2X = (oldDelta2X + zeroPointerVec2PosX(curRow + windowM, 0) + 1 + histSizeX) % histSizeX;
			int newIdx1X = (newDelta1X + zeroPointerVec1PosX(curRow, 0) + histSizeX) % histSizeX;
			int newIdx2X = (newDelta2X + zeroPointerVec2PosX(curRow + windowM, 0) + histSizeX) % histSizeX;
			// old values
			histMat1PosX(curRow, oldIdx1X) = histMat1PosX(curRow, oldIdx1X) - oldWeight1X;
			histMat2PosX(curRow + windowM, oldIdx2X) = histMat2PosX(curRow + windowM, oldIdx2X) - oldWeight2X;
			// new values
			histMat1PosX(curRow, newIdx1X) = histMat1PosX(curRow, newIdx1X) + newWeight1X;
			histMat2PosX(curRow + windowM, newIdx2X) = histMat2PosX(curRow + windowM, newIdx2X) + newWeight2X;

			dSumVec1PosX(curRow, 0) = (exp1(-1))*(dSumVec1PosX(curRow, 0) - e11X) + histMat1PosX(curRow, zeroPointerVec1PosX(curRow, 0));
			dSumVec2PosX(curRow + windowM, 0) = (exp1(-1))*(dSumVec2PosX(curRow + windowM, 0) - e21X) + histMat2PosX(curRow + windowM, zeroPointerVec2PosX(curRow + windowM, 0));


			// NEGATIVE

			int negCurColX = w2 - w - (k - 1);

			delta1NegX = ourSumVecNegX(curRow, negCurColX - 1);
			delta2NegX = ourSumVecNegX(curRow, negCurColX);
			delta3NegX = ourSumVecNegX(curRow + 1, negCurColX);
			// handle new and old values

			oldDelta1NegX = xyPositions(xyPositionIdx, nnfMat(curRow, negCurColX + w - 1)) - (w - 1);
			oldWeight1NegX = invFreqMat(curRow, negCurColX + w - 1);
			if (oldDelta1NegX == 0) {
				e12NegX = 0;
			}
			else {
				e12NegX = oldWeight1NegX*exp1(-abs(oldDelta1NegX));
			}

			oldDelta2NegX = xyPositions(xyPositionIdx, nnfMat(curRow + windowM, negCurColX + w - 1)) - (w - 1);
			oldWeight2NegX = invFreqMat(curRow + windowM, negCurColX + w - 1);
			if (oldDelta2NegX == 0) {
				e22NegX = 0;
			}
			else {
				e22NegX = oldWeight2NegX*exp1(-abs(oldDelta2NegX));
			}
			newDelta1NegX = xyPositions(xyPositionIdx, nnfMat(curRow, negCurColX - 1)) - 0;
			newWeight1NegX = invFreqMat(curRow, negCurColX - 1);
			newDelta2NegX = xyPositions(xyPositionIdx, nnfMat(curRow + windowM, negCurColX - 1)) - 0;
			newWeight2NegX = invFreqMat(curRow + windowM, negCurColX - 1);

			// increment the zero pointer
			if (zeroPointerVec1NegX(curRow, 0) == histSizeX - 1) {
				zeroPointerVec1NegX(curRow, 0) = 0;
			}
			else {
				zeroPointerVec1NegX(curRow, 0) = zeroPointerVec1NegX(curRow, 0) + 1;
			}

			if (zeroPointerVec2NegX(curRow + windowM, 0) == histSizeX - 1) {
				zeroPointerVec2NegX(curRow + windowM, 0) = 0;
			}
			else {
				zeroPointerVec2NegX(curRow + windowM, 0) = zeroPointerVec2NegX(curRow + windowM, 0) + 1;
			}

			// update the histograms
			// old values
			oldIdx1X = (oldDelta1NegX + zeroPointerVec1NegX(curRow, 0) - 1 + histSizeX) % histSizeX;
			histMat1NegX(curRow, oldIdx1X) = histMat1NegX(curRow, oldIdx1X) - oldWeight1NegX;
			oldIdx2X = (oldDelta2NegX + zeroPointerVec2NegX(curRow + windowM, 0) - 1 + histSizeX) % histSizeX;
			histMat2NegX(curRow + windowM, oldIdx2X) = histMat2NegX(curRow + windowM, oldIdx2X) - oldWeight2NegX;
			int minusOneIdx1X = (zeroPointerVec1NegX(curRow, 0) - 1 + histSizeX) % histSizeX;
			firstRowDiffX = (exp1(-1))*(dSumVec1NegX(curRow, 0) - e12NegX) + histMat1NegX(curRow, minusOneIdx1X)*(exp1(-1));
			float firstRowDiffXTag = dSumVec1NegX(curRow, 0);
			int minusOneIdx2X = (zeroPointerVec2NegX(curRow + windowM, 0) - 1 + histSizeX) % histSizeX;
			lastRowDiffX = (exp1(-1))*(dSumVec2NegX(curRow + windowM, 0) - e22NegX) + histMat2NegX(curRow + windowM, minusOneIdx2X)*(exp1(-1));
			float lastRowDiffXTag = dSumVec2NegX(curRow + windowM, 0);
			ourSumVecNegX(curRow + 1, negCurColX - 1) = delta1NegX + delta3NegX - delta2NegX + (lastRowDiffX - lastRowDiffXTag) - (firstRowDiffX - firstRowDiffXTag);

			// new values
			newIdx1X = (newDelta1NegX + zeroPointerVec1NegX(curRow, 0) + histSizeX) % histSizeX;
			histMat1NegX(curRow, newIdx1X) = histMat1NegX(curRow, newIdx1X) + newWeight1NegX;
			newIdx2X = (newDelta2NegX + zeroPointerVec2NegX(curRow + windowM, 0) + histSizeX) % histSizeX;
			histMat2NegX(curRow + windowM, newIdx2X) = histMat2NegX(curRow + windowM, newIdx2X) + newWeight2NegX;

			dSumVec1NegX(curRow, 0) = (exp1(-1))*(dSumVec1NegX(curRow, 0) - e12NegX) + histMat1NegX(curRow, minusOneIdx1X)*(exp1(-1));
			dSumVec2NegX(curRow + windowM, 0) = (exp1(-1))*(dSumVec2NegX(curRow + windowM, 0) - e22NegX) + histMat2NegX(curRow + windowM, minusOneIdx2X)*(exp1(-1));

		}

	}

	for (int i = 0; i <= rowCount - windowM; i++) {
		for (int j = 0; j <= colCount - windowN; j++) {
			scoreMap_out(i, j) = ourSumVecPosX(i, j) + ourSumVecNegX(i, j);
		}
	}
}

void InitForOneDimExpL1MultiSide(Mat<float> & dSumVec1Pos, Mat<float> & histMat1Pos, Mat<float> & dSumVec1Neg, Mat<float> & histMat1Neg, Mat<int> nnfMat, Mat<float> invFreqMat, const Map<MatrixXi>& xyPositions, int windowN, int curRow, int xyPositionIdx) {

	/// X
	float curRowSumDiffNegX = 0.0;
	float curRowSumDiffPosX = 0.0;
	int curDiffX, idxX, jRev;
	int colCount = nnfMat.width;

	for (int j = 0; j < windowN; j++) {
		// X
		curDiffX = xyPositions(xyPositionIdx, nnfMat(curRow, j)) - j;
		if (curDiffX >= 0) {
			curRowSumDiffPosX = curRowSumDiffPosX + invFreqMat(curRow, j)*exp1(-abs(curDiffX));
		}

		idxX = curDiffX + windowN;
		histMat1Pos(idxX, 0) = histMat1Pos(idxX, 0) + invFreqMat(curRow, j);

		jRev = colCount - windowN + j;
		curDiffX = xyPositions(xyPositionIdx, nnfMat(curRow, jRev)) - (jRev - (colCount - windowN));
		if (curDiffX < 0) {
			curRowSumDiffNegX = curRowSumDiffNegX + invFreqMat(curRow, jRev)*exp1(-abs(curDiffX));
		}
		idxX = curDiffX + windowN;
		histMat1Neg(idxX, 0) = histMat1Neg(idxX, 0) + invFreqMat(curRow, jRev);

	}

	dSumVec1Pos(0, 0) = curRowSumDiffPosX;
	dSumVec1Neg(colCount-windowN, 0) = curRowSumDiffNegX;

}

void OneDimExpL1DistWeightedMultiSide(Mat<int> nnfMat, Mat<float> invFreqMat, int windowM, int windowN, const Map<MatrixXi>& xyPositions, Mat<float>& scoreMap_out, int curRow, int xyPositionIdx) {
	
	int rowCount = nnfMat.height;
	int colCount = nnfMat.width;
	int h = windowM, w = windowN, oldDelta1X, newDelta1X, oldDelta1NegX, newDelta1NegX;
	int histSizeX = 2 * w + 1;
	int curCol;
	float oldWeight1X, e11X, newWeight1X, e12X;
	int zeroPointer1PosX = w;
	Mat<float> dSumVec1PosX = CreateZeroMatFloat(colCount, 1);
	Mat<float> histMat1PosX = CreateZeroMatFloat(histSizeX, 1);

	float oldWeight1NegX, e12NegX, newWeight1NegX;
	int zeroPointer1NegX = w;
	Mat<float> dSumVec1NegX = CreateZeroMatFloat(colCount, 1);
	Mat<float> histMat1NegX = CreateZeroMatFloat(histSizeX, 1);

	InitForOneDimExpL1MultiSide(dSumVec1PosX, histMat1PosX, dSumVec1NegX, histMat1NegX, nnfMat, invFreqMat, xyPositions, windowN, curRow, xyPositionIdx);
	scoreMap_out(curRow, 0) = dSumVec1PosX(0, 0);
	scoreMap_out(curRow, colCount - windowN) = dSumVec1NegX(colCount - windowN, 0);

	for (int i = 1; i < colCount - windowN; i++) {
		curCol = i - 1;
		// POSITIVE
		oldDelta1X = xyPositions(xyPositionIdx, nnfMat(curRow, curCol)) - 0;
		oldWeight1X = invFreqMat(curRow, curCol);
		e11X = oldWeight1X*exp1(-oldDelta1X);

		newDelta1X = xyPositions(xyPositionIdx, nnfMat(curRow, curCol + w)) - (w - 1);
		newWeight1X = invFreqMat(curRow, curCol + w);
		e12X = abs((float)newDelta1X);
		if (newDelta1X == 0) {
			e12X = newWeight1X;
		}
		else {
			e12X = 0;
		}

		zeroPointer1PosX = (zeroPointer1PosX - 1 + histSizeX) % (histSizeX);

		int oldIdx1X = (oldDelta1X + zeroPointer1PosX + 1 + histSizeX) % histSizeX;
		int newIdx1X = (newDelta1X + zeroPointer1PosX + histSizeX) % histSizeX;

		// old values
		histMat1PosX(oldIdx1X, 0) = histMat1PosX(oldIdx1X, 0) - oldWeight1X;
		histMat1PosX(newIdx1X, 0) = histMat1PosX(newIdx1X, 0) + newWeight1X;

		dSumVec1PosX(i, 0) = (exp1(-1))*(dSumVec1PosX(curCol, 0) - e11X) + histMat1PosX(zeroPointer1PosX, 0);

		// NEGATIVE

		int negCurColX = colCount - w - (i - 1);

		oldDelta1NegX = xyPositions(xyPositionIdx, nnfMat(curRow, negCurColX + w - 1)) - (w - 1);
		oldWeight1NegX = invFreqMat(curRow, negCurColX + w - 1);
		if (oldDelta1NegX == 0) {
			e12NegX = 0;
		}
		else {
			e12NegX = oldWeight1NegX*exp1(-abs(oldDelta1NegX));
		}

		newDelta1NegX = xyPositions(xyPositionIdx, nnfMat(curRow, negCurColX - 1)) - 0;
		newWeight1NegX = invFreqMat(curRow, negCurColX - 1);

		// increment the zero pointer
		if (zeroPointer1NegX == histSizeX - 1) {
			zeroPointer1NegX = 0;
		}
		else {
			zeroPointer1NegX = zeroPointer1NegX + 1;
		}

		// update the histograms
		// old values
		oldIdx1X = (oldDelta1NegX + zeroPointer1NegX - 1 + histSizeX) % histSizeX;
		histMat1NegX(oldIdx1X, 0) = histMat1NegX(oldIdx1X, 0) - oldWeight1NegX;

		int minusOneIdx1X = (zeroPointer1NegX - 1 + histSizeX) % histSizeX;

		// new values
		newIdx1X = (newDelta1NegX + zeroPointer1NegX + histSizeX) % histSizeX;
		histMat1NegX(newIdx1X, 0) = histMat1NegX(newIdx1X, 0) + newWeight1NegX;

		dSumVec1NegX(negCurColX-1, 0) = (exp1(-1))*(dSumVec1NegX(negCurColX, 0) - e12NegX) + histMat1NegX(minusOneIdx1X, 0)*(exp1(-1));

		scoreMap_out(curRow, i) = scoreMap_out(curRow, i) + dSumVec1PosX(i, 0);
		scoreMap_out(curRow, negCurColX - 1) = scoreMap_out(curRow, negCurColX - 1) + dSumVec1NegX(negCurColX - 1, 0);
	}
}

void conv2(const cv::Mat &img, const cv::Mat & kernel, ConvolutionType type, cv::Mat & dest) {
	cv::Mat source = img;

	cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = cv::BORDER_CONSTANT;
	cv::Mat result = kernel;
	cv::flip(kernel, result, -1);
	filter2D(source, dest, img.depth(), result , anchor, 0, borderMode);

	if (CONVOLUTION_VALID == type) {
		dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2);
		dest = dest.rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	}
}

void DIWUIntegralImage(Mat<int> nnfMat, Mat<float> invFreqMat, int windowM, int windowN, const Map<MatrixXi>& xyPositions, Mat<float>& scoreMap_out, bool isX) {


	int rowCount = nnfMat.height;
	int colCount = nnfMat.width;
	Mat<float> midMapScores = CreateZeroMatFloat(rowCount, colCount - windowN + 1);
	int xyPositionIdx;
	if (isX) {
		xyPositionIdx = 0;
	}
	else {
		xyPositionIdx = 1;
	}

	for (int i = 0; i < rowCount; i++) {
		OneDimExpL1DistWeightedMultiSide(nnfMat, invFreqMat, windowM, windowN, xyPositions, midMapScores, i, xyPositionIdx);
	}


	cv::Mat kernel(windowM, 1, CV_32F);
	for (int i = 0; i < windowM; i++) {
		kernel.at<float>(i, 0) = 1;
	}

	cv::Mat inScoreMapCV = toCVMatFloat(midMapScores);
	cv::Mat outScoreMapCV = toCVMatFloat(midMapScores);
	conv2(inScoreMapCV, kernel, ConvolutionType::CONVOLUTION_VALID, outScoreMapCV);

	for (int i = 0; i < outScoreMapCV.rows; i++) {
		for (int j = 0; j < outScoreMapCV.cols; j++) {
			scoreMap_out(i, j) = outScoreMapCV.at<float>(i, j);
		}
	}

}

void saveIm(cv::Mat image) {
	cv::imwrite("im.jpg", image);
	cv::imshow("image", image);
	cv::waitKey(0);
}

void DIWURasterScan(Mat<int> nnfMat, Mat<float> invFreqMat, int windowM, int windowN, const Map<MatrixXi>& xyPositions, Mat<float>& scoreMap_outX, Mat<float>& scoreMap_outY)
{

	DIWURasterScanX(nnfMat, invFreqMat, windowM, windowN, xyPositions, scoreMap_outX, true);

	// transpose nnfMat and invFreqMat
	Mat<int> nnfMatTranspose = nnfMat.transpose();
	Mat<float> invFreqMatTranspose = invFreqMat.transpose();
	DIWURasterScanX(nnfMatTranspose, invFreqMatTranspose, windowN, windowM, xyPositions, scoreMap_outY, false);

	Mat<float> scoreMap_outYTranspose = scoreMap_outY.transpose();
	for (int i = 0; i < scoreMap_outX.height; i++) {
		for (int j = 0; j < scoreMap_outX.width; j++) {
			scoreMap_outX(i, j) = scoreMap_outX(i, j) + scoreMap_outYTranspose(i, j);
		}
	}
}

void IWURasterScan_ThreadFunc(iwuScanParams p)
{
	IWURasterScan(*p.pNnfMat, *p.invFreqMat, p.windowM, p.windowN, *p.pXyPositions, p.h, *p.pScoreMap, p.startColIndexInclusive, p.endColIndexExclusive, &p.maxElement);
}

void DIWURasterScan_ThreadFunc(diwuScanParams p)
{

	DIWUIntegralImage(*p.pNnfMat, *p.invFreqMat, p.windowM, p.windowN, *p.pXyPositions, *p.pScoreMapX, true);
	
	// transpose nnfMat and invFreqMat
	Mat<int> nnfMat = *p.pNnfMat;
	Mat<int> nnfMatTranspose = nnfMat.transpose();
	Mat<float> invFreqMat = *p.invFreqMat;
	Mat<float> invFreqMatTranspose = invFreqMat.transpose();
	Mat<float>& scoreMap_outX = *p.pScoreMapX;
	Mat<float>& scoreMap_outY = *p.pScoreMapY;
	DIWUIntegralImage(nnfMatTranspose, invFreqMatTranspose, p.windowN, p.windowM, *p.pXyPositions, *p.pScoreMapY, false);

	Mat<float> scoreMap_outYTranspose = scoreMap_outY.transpose();
	for (int i = 0; i < scoreMap_outX.height; i++) {
		for (int j = 0; j < scoreMap_outX.width; j++) {
			scoreMap_outX(i, j) = scoreMap_outX(i, j) + scoreMap_outYTranspose(i, j);
		}
	}
	
}

Map<MatrixXi> LoadXyPositions(const mxArray** inputs)
{
	MxArray mxXYPositions = MxArray(inputs[Idx::IN_xyPositons]);
	auto xyPositions = mexArray2EigenMat_int(mxXYPositions);

	ASSERT2(
		xyPositions.cols() == 2 || xyPositions.rows() == 2,
		"argument xyPositions must be a matrix of (2 X N) where N is the amount of available patches in the database. N must be greater than the maximum index of in the nnf matrix");

	if (xyPositions.cols() == 2)
	{
		xyPositions.transposeInPlace();
	}

	return xyPositions;
}

void diwu::IWU(mxArray** outputs, int inputsElementCount, const mxArray** inputs, unsigned nthreads)
{
	if (nthreads <= 0)
	{
		nthreads = 1;
	}

	/*  loading Nearest neighbor field map */
	int windowM, windowN;
	Mat<int> nnfMat = ExtractNnfAndWindow(inputs, windowM, windowN);
	Mat<float> invFreqMat = ExtractInvFreq(inputs);
	int nnfM = nnfMat.height;
	int nnfN = nnfMat.width;
	const int* nnfArray = &nnfMat(0, 0);
	ASSERT2(inputsElementCount >= Idx::IN_xyPositons + 1, "missing argument: xy positins matrix");

	Map<MatrixXi> xyPositions = LoadXyPositions(inputs);

	double h = 1;
	if (inputsElementCount >= Idx::IN_h + 1)
	{
		h = MxArray(inputs[Idx::IN_h]).toDouble();
	}

	Mat<float> diversityScoreMap = CreateOutputScoreMap(outputs, windowM, windowN, nnfM, nnfN);

	// Seperate work between threads, building parameters.
	std::vector<iwuScanParams> paramsVec(nthreads);
	std::vector<thread> threadVec(nthreads - 1);  // minus one since using the current thread
	int outputN = diversityScoreMap.width;

	int chunkSize = outputN / nthreads;
	int leftoverModulu = outputN % nthreads;
	int startRangeInclusive = 0;
	int maxElement = FindMaxNNIndex(nnfMat);
	for (int i = 0; i < paramsVec.size(); i++)
	{
		int endRangeExclusive = startRangeInclusive + chunkSize;
		if (leftoverModulu > 0)
		{
			++endRangeExclusive;
			--leftoverModulu;
		}

		paramsVec[i] = iwuScanParams(
			startRangeInclusive, endRangeExclusive, windowM, windowN, h,
			&xyPositions, &diversityScoreMap, &nnfMat, &invFreqMat, maxElement);

		startRangeInclusive = endRangeExclusive;
	}

	//Launch threads 1 to NTHREADS-1
	for (int threadInd = 0; threadInd < threadVec.size(); threadInd++)
	{
		threadVec.at(threadInd) = thread(IWURasterScan_ThreadFunc, std::ref(paramsVec[threadInd]));
	}

	//using the current thread as the last thread, updating a chunk of old patches
	IWURasterScan_ThreadFunc(paramsVec[nthreads - 1]);

	//Wait for all threads to finish
	for (int threadInd = 0; threadInd < threadVec.size(); ++threadInd)
	{
		threadVec[threadInd].join();
	}
}

void diwu::DIWU(mxArray** outputs, int inputsElementCount, const mxArray** inputs, unsigned nthreads)
{

	nthreads = 1;
	/*  loading Nearest neighbor field map */
	int windowM, windowN;
	Mat<int> nnfMat = ExtractNnfAndWindow(inputs, windowM, windowN);
	Mat<float> invFreqMat = ExtractInvFreq(inputs);
	int nnfM = nnfMat.height;
	int nnfN = nnfMat.width;
	const int* nnfArray = &nnfMat(0, 0);

	Map<MatrixXi> xyPositions = LoadXyPositions(inputs);

	Mat<float> diversityScoreMapX = CreateOutputScoreMap(outputs, windowM, windowN, nnfM, nnfN);
	Mat<float> diversityScoreMapY = diversityScoreMapX.transpose();

	std::vector<diwuScanParams> paramsVec(nthreads);
	std::vector<thread> threadVec(nthreads - 1);  // minus one since using the current thread
	int outputN = diversityScoreMapX.width;


	paramsVec[0] = diwuScanParams(windowM, windowN, &xyPositions, &diversityScoreMapX, &diversityScoreMapY, &nnfMat, &invFreqMat);

	//Launch threads 1 to NTHREADS-1
	for (int threadInd = 0; threadInd < threadVec.size(); threadInd++)
	{
		threadVec.at(threadInd) = thread(DIWURasterScan_ThreadFunc, std::ref(paramsVec[threadInd]));
	}

	//using the current thread as the last thread, updating a chunk of old patches
	DIWURasterScan_ThreadFunc(paramsVec[nthreads - 1]);

	//Wait for all threads to finish
	for (int threadInd = 0; threadInd < threadVec.size(); ++threadInd)
	{
		threadVec[threadInd].join();
	}
}


