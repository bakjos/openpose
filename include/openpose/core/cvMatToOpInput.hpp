#ifndef OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "array.hpp"
#include "gpuArray.hpp"

namespace op
{
    class OPENPOSE_API CvMatToOpInput
    {
    public:
        CvMatToOpInput(const cv::Size& netInputResolution, const int scaleNumber = 1, const float scaleGap = 0.25);

        Array<float> format(const cv::Mat& cvInputData) const;

		void	 format(GpuArray<float>& gpuArray, const cv::cuda::GpuMat& cvInputData) const;

		~CvMatToOpInput();

    private:
        const int mScaleNumber;
        const float mScaleGap;
        const std::vector<int> mInputNetSize4D;
		cv::cuda::GpuMat*		scaledInputData;
    };
}

#endif // OPENPOSE__CORE__CV_MAT_TO_OP_INPUT_HPP
