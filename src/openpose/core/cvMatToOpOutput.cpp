#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/utilities/openCv.hpp"
#include "openpose/core/cvMatToOpOutput.hpp"

namespace op
{
    CvMatToOpOutput::CvMatToOpOutput(const cv::Size& outputResolution, const bool generateOutput) :
        mGenerateOutput{generateOutput},
        mOutputSize3D{{3, outputResolution.height, outputResolution.width}},
		scaledImage {new cv::cuda::GpuMat()}
    {
    }

	CvMatToOpOutput::~CvMatToOpOutput() {
		delete scaledImage;
    }

    std::tuple<double, Array<float>> CvMatToOpOutput::format(const cv::Mat& cvInputData) const
    {
        try
        {
            // Security checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);

            // outputData - Reescale keeping aspect ratio and transform to float the output image
            const cv::Size outputResolution{mOutputSize3D[2], mOutputSize3D[1]};
            const double scaleInputToOutput = resizeGetScaleFactor(cvInputData.size(), outputResolution);
            const cv::Mat frameWithOutputSize = resizeFixedAspectRatio(cvInputData, scaleInputToOutput, outputResolution);
            Array<float> outputData;
            if (mGenerateOutput)
            {
                outputData.reset(mOutputSize3D);
                uCharCvMatToFloatPtr(outputData.getPtr(), frameWithOutputSize, false);
            }

            return std::make_tuple(scaleInputToOutput, outputData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(0., Array<float>{});
        }
    }

	void  CvMatToOpOutput::format(const cv::cuda::GpuMat& cvInputData, double& scaleInputToOutput, GpuArray<float>& outputData) const {
		try
		{
			// Security checks
			if (cvInputData.empty())
				error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);

			// outputData - Reescale keeping aspect ratio and transform to float the output image
			const cv::Size outputResolution{ mOutputSize3D[2], mOutputSize3D[1] };
			scaleInputToOutput = resizeGetScaleFactor(cvInputData.size(), outputResolution);
			resizeFixedAspectRatioGpu(cvInputData, *scaledImage, scaleInputToOutput, outputResolution);
			if ( outputData.empty() ) {
				outputData.reset(mOutputSize3D);
			}

			if (mGenerateOutput)
			{	
				uCharGpuMatToFloatPtr(outputData.getPtr(), *scaledImage, false);
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			scaleInputToOutput = 0;
		}
    }
}
