#ifndef OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "array.hpp"
#include "gpuArray.hpp"

namespace op
{
    class OPENPOSE_API OpOutputToCvMat
    {
    public:
        explicit OpOutputToCvMat(const cv::Size& outputResolution);

        cv::Mat formatToCvMat(const Array<float>& outputData) const;

		void formatToCvMat(const GpuArray<float>& outputData, cv::cuda::GpuMat& cvMat) const;

    private:
        const cv::Size mOutputResolution;
    };
}

#endif // OPENPOSE__CORE__OP_OUTPUT_TO_CV_MAT_HPP
