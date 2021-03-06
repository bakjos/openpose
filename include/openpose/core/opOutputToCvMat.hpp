#ifndef OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "array.hpp"
#include "point.hpp"
#include "gpuArray.hpp"

namespace op
{
    class OPENPOSE_API OpOutputToCvMat
    {
    public:
        explicit OpOutputToCvMat(const Point<int>& outputResolution);

        cv::Mat formatToCvMat(const Array<float>& outputData) const;

		void formatToCvMat(const GpuArray<float>& outputData, cv::cuda::GpuMat& cvMat) const;

    private:
        const Point<int> mOutputResolution;
    };
}

#endif // OPENPOSE_CORE_OP_OUTPUT_TO_CV_MAT_HPP
