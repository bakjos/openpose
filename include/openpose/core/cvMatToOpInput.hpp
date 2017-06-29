#ifndef OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
#define OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP

#include <utility> // std::pair
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "array.hpp"
#include "point.hpp"
#include "gpuArray.hpp"

namespace op
{
    class OPENPOSE_API CvMatToOpInput
    {
    public:
        CvMatToOpInput(const Point<int>& netInputResolution, const int scaleNumber = 1, const float scaleGap = 0.25);

        std::pair<Array<float>, std::vector<float>> format(const cv::Mat& cvInputData) const;

		std::vector<float> format(GpuArray<float>& gpuArray, const cv::cuda::GpuMat& cvInputData) const;

		~CvMatToOpInput();

    private:
        const int mScaleNumber;
        const float mScaleGap;
        const std::vector<int> mInputNetSize4D;
		cv::cuda::GpuMat*		scaledInputData;
    };
}

#endif // OPENPOSE_CORE_CV_MAT_TO_OP_INPUT_HPP
