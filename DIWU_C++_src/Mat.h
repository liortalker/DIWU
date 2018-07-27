#pragma once

namespace diwu
{
	template <typename T>
	class Mat
	{
		T* innerArray;
	public:
		
		Mat(T* data, int w, int h) : 
			innerArray(data),
			width(w),
			height(h)
		{}

		const int width, height;

		int sub2ind(int i, int j) const
		{
			int result = i + j*height;
			return result;
		}

		const T& operator()(int i, int j) const
		{
			return innerArray[sub2ind(i, j)];
		}

		T& operator()(int i, int j)
		{
			if (i < 0 || i >= height || j < 0 || j >= width) {
				cout << i << j;
			}

			return innerArray[sub2ind(i, j)];
		}

		Mat<T> transpose() {

			T* transposeData = new T[width*height];
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					transposeData[i*width + j] = innerArray[i + j*height];
				}
			}
			return Mat<T>(transposeData, height, width);
		}

		/*cv::Mat toCVMat() {
			return cv::Mat(height, width, CV_32F, innerArray);
		}*/

	};
}