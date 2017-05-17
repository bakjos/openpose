#ifndef OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP
#define OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "array.hpp"
#include "gpuArray.hpp"

namespace op
{
    class OPENPOSE_API CvMatToOpOutput
    {
    public:
        CvMatToOpOutput(const cv::Size& outputResolution, const bool generateOutput = true);
		~CvMatToOpOutput();

        std::tuple<double, Array<float>> format(const cv::Mat& cvInputData) const;

		void format(const cv::cuda::GpuMat& cvInputData, double& scaleInputToOutput, GpuArray<float>& outputData) const;

    private:
        const bool mGenerateOutput;
        const std::vector<int> mOutputSize3D;
		cv::cuda::GpuMat* scaledImage;
    };
}

#endif // OPENPOSE__CORE__CV_MAT_TO_OP_OUTPUT_HPP
