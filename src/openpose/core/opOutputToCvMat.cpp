#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/core/opOutputToCvMat.hpp>

namespace op
{
    OpOutputToCvMat::OpOutputToCvMat(const Point<int>& outputResolution) :
        mOutputResolution{outputResolution}
    {
    }

    cv::Mat OpOutputToCvMat::formatToCvMat(const Array<float>& outputData) const
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);

            cv::Mat cvMat;
            floatPtrToUCharCvMat(cvMat, outputData.getConstPtr(), mOutputResolution, 3);

            return cvMat;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat{};
        }
    }

	void OpOutputToCvMat::formatToCvMat(const GpuArray<float>& outputData, cv::cuda::GpuMat& cvMat) const {
		try
		{
			// Security checks
			if (outputData.empty())
				error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);

			floatPtrToUCharGpuMat(cvMat, outputData.getConstPtr(), mOutputResolution, 3);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
    }
}
