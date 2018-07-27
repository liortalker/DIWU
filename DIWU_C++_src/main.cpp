#include "main.h"
#include "mex.h"
#include "../mex_opencv_2.4/MxArray.hpp"
#include "../mex_opencv_2.4/EigenExtensions.h"
#include "mexInterface.h"
#include "DIWU.h"
#include "tictoc.h"

using namespace std;
using namespace diwu;

struct MethodTypes {
	static const char* IWU;
	static const char* DIWU;
	static vector<const char*> All;
};

const char* MethodTypes::IWU = "IWU";
const char* MethodTypes::DIWU = "DIWU";
vector<const char*> MethodTypes::All = { IWU, DIWU };


template <typename T>
string stringJoin(string delim, vector<T> objs)
{

	if (objs.size() <= 0) return "";
	std::string result = objs[0];
	for (int i = 1; i < objs.size(); ++i)
	{
		T o = objs[i];
		result += delim;
		result += o;
	}

	return result;
}

string excapsulate(string enc, string str)
{
	return enc + str + enc;
}

string BuildInputErrMessage()
{
	ostringstream stringStream;
	stringStream
		<< "Expected inputs: " << "\n"
		<< excapsulate("'", MethodTypes::IWU) << ", problem with the input" << "\n"
		<< excapsulate("'", MethodTypes::DIWU) << ", problem with the input" << "\n";
		
	string message = stringStream.str().c_str();
	return message;
}


void finishWithTimiings(double secs)
{
	ostringstream s;
	s << "elapsed time in seconds: " << secs << "\n";
	mexPrintf(s.str().c_str());
}

// main gate to matlab
void mexFunction(int outputElementCount, mxArray *outputs[], int inputsElementCount, const mxArray *inputs[])
{
	try
	{
		// verify signature of function
		ASSERT2(
			inputsElementCount >= Idx::IN_windowSizeN + 1, 
			BuildInputErrMessage().c_str());

		ASSERT2(
			outputElementCount == 1,
			"expected outpus: (diversity map)");

		string methodType;
		try
		{
			methodType = MxArray(inputs[Idx::IN_DIVERSITY_TYPE]).toString();
		}
		catch (...)
		{
			ERROR((string("First parameter should be a string. ") + BuildInputErrMessage()).c_str());
		}

		if (methodType == MethodTypes::IWU)
		{
			IWU(outputs, inputsElementCount, inputs);
			return;
		}

		if (methodType == MethodTypes::DIWU)
		{
			DIWU(outputs, inputsElementCount, inputs);
			return;
		}

		string delim = "/";
		std::string errMessage = "first parameter should be one of the following options: ";
		errMessage += stringJoin(delim, MethodTypes::All);
		ERROR(errMessage.c_str());

		auto nnfMat2 = mexArray2EigenMat_int(MxArray(inputs[Idx::IN_NNF]));

	}
	catch (int e)
	{
		ostringstream s;
		s << "An exception occurred. Exception Nr. " << e << '\n';
		ERROR(s.str().c_str());
	}
}

