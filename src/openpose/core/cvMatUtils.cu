#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "openpose/utilities/cuda.hpp"
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/cuda.hpp"

namespace op {

	//float* (deep net format) : C x H x W
	//cv::Mat(OpenCV format) : H x W x C

	__global__ void gpuMatToFloatKernel(float* floatImagePtr, const uchar* imgData, int channels, unsigned int width, unsigned int height, int step, bool normalize, unsigned long offset) {
		float* floatImage = floatImagePtr + offset;
		const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= width || y >= height)
		{
			return;
		}

		
		const auto originFramePtrOffset = (y*step) + (x*channels);

		for (auto c = 0; c < channels; c++)
		{
			const auto floatImageOffsetC = c * height;
			const auto floatImageOffsetY = (floatImageOffsetC + y) * width;
			auto val = float(imgData[originFramePtrOffset + c]);
			if (normalize) {
				val = (val /256.f) - 0.5f;
			}
			floatImage[floatImageOffsetY + x] = val;
		}
		
	}

	__global__ void floatTogpuMatKernel(uchar* imgData, const float* floatImage, int channels, unsigned int width, unsigned int height, int step) {
		const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= width || y >= height)
		{
			return;
		}

		const auto offsetBetweenChannels = width * height;
		const auto cvMatOffset = (y*step) + (x*channels);
		for (auto c = 0; c < channels; c++)
		{
			const auto offsetChannelC = c*offsetBetweenChannels;
			const auto floatImageOffsetY = offsetChannelC + y*width;
			const auto value = uchar(__float2int_rn(floatImage[floatImageOffsetY + x]));
			imgData[cvMatOffset + c] = value;
		}

	}
	
	void gpuMatToFloatPtr(float* floatImage, const unsigned char* imgData, const int channels, const cv::Size& sourceSize, const size_t step, const bool normalize, const unsigned long offset) {
		dim3 threadsPerBlock;
		dim3 numBlocks;
		std::tie(threadsPerBlock, numBlocks) = getNumberCudaThreadsAndBlocks(sourceSize);		

		gpuMatToFloatKernel << <threadsPerBlock, numBlocks >> > (floatImage, imgData, channels, sourceSize.width, sourceSize.height, step, normalize, offset);
		cudaCheck(__LINE__, __FUNCTION__, __FILE__);
	}

	void floatPtrToGpuMat(unsigned char* imgData, const float* floatImage, const int channels, const cv::Size& sourceSize, const size_t step) {
		dim3 threadsPerBlock;
		dim3 numBlocks;
		std::tie(threadsPerBlock, numBlocks) = getNumberCudaThreadsAndBlocks(sourceSize);
		floatTogpuMatKernel << <threadsPerBlock, numBlocks >> > (imgData, floatImage, channels, sourceSize.width, sourceSize.height, step);
	}

}