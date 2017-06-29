#ifndef OPENPOSE_UTILITIES_OPEN_CV_HPP
#define OPENPOSE_UTILITIES_OPEN_CV_HPP

#include <opencv2/core/core.hpp> // cv::Mat, cv::Point
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/core/core.hpp> // cv::Mat, cv::Point
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc/imgproc.hpp> // cv::warpAffine, cv::BORDER_CONSTANT
#include <openpose/core/point.hpp> // Point<int>

namespace op
{
    
    
	OPENPOSE_API void putTextOnCvMat(cv::Mat& cvMat, const std::string& textToDisplay, const Point<int>& position, const cv::Scalar& color, const bool normalizeWidth);

	OPENPOSE_API void floatPtrToUCharCvMat(cv::Mat& cvMat, const float* const floatImage, const Point<int>& resolutionSize, const int resolutionChannels, int cvMatOffsetY = 0);

	OPENPOSE_API void floatPtrToUCharGpuMat(cv::cuda::GpuMat& cvMat, const float* const floatImage, const Point<int>& resolutionSize, const int resolutionChannels);

 
	OPENPOSE_API void unrollArrayToUCharCvMat(cv::Mat& cvMatResult, const Array<float>& array);

	OPENPOSE_API void uCharCvMatToFloatPtr(float* floatImage, const cv::Mat& cvImage, const bool normalize);

	OPENPOSE_API void uCharGpuMatToFloatPtr(float* floatImage, const cv::cuda::GpuMat& cvImage, const bool normalize, const unsigned long offset = 0);
   
	OPENPOSE_API double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize);

    
   
	OPENPOSE_API  cv::Mat resizeFixedAspectRatio(const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize, const int borderMode = cv::BORDER_CONSTANT,
                                   const cv::Scalar& borderValue = cv::Scalar{0,0,0});

	OPENPOSE_API void resizeFixedAspectRatioGpu(const cv::cuda::GpuMat& cvMat, cv::cuda::GpuMat& dst, const double scaleFactor, const Point<int>& targetSize, const int borderMode = cv::BORDER_CONSTANT,
		const cv::Scalar& borderValue = cv::Scalar{ 0,0,0 });

	OPENPOSE_API void gpuMatToFloatPtr(float* floatImage, const unsigned char* imgData, const int channels, const Point<int>& sourceSize, const size_t step, const bool normalize, const unsigned long offset);

	OPENPOSE_API void floatPtrToGpuMat(unsigned char* imgData, const float* floatImage, const int channels, const Point<int>& sourceSize, const size_t step);
}

#endif // OPENPOSE_UTILITIES_OPEN_CV_HPP
