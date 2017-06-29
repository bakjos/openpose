#ifndef OPENPOSE_FILESTREAM_FILE_STREAM_HPP
#define OPENPOSE_FILESTREAM_FILE_STREAM_HPP

#include <openpose/config.hpp>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // CV_LOAD_IMAGE_ANYDEPTH, CV_IMWRITE_PNG_COMPRESSION
#include <openpose/core/array.hpp>
#include "enumClasses.hpp"



namespace op
{
	OPENPOSE_API DataFormat stringToDataFormat(const std::string& dataFormat);

    // Save/load json, xml, yaml, yml
	OPENPOSE_API void saveData(const std::vector<cv::Mat>& cvMats, const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

	OPENPOSE_API void saveData(const cv::Mat& cvMat, const std::string cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

	OPENPOSE_API std::vector<cv::Mat> loadData(const std::vector<std::string>& cvMatNames, const std::string& fileNameNoExtension, const DataFormat format);

	OPENPOSE_API cv::Mat loadData(const std::string& cvMatName, const std::string& fileNameNoExtension, const DataFormat format);

    // Json - Saving as *.json not available in OpenCV verions < 3.0, this function is a quick fix
	OPENPOSE_API void saveKeypointsJson(const Array<float>& pose, const std::string& fileName, const bool humanReadable, const std::string& keypointName);

    // Save/load image
	OPENPOSE_API void saveImage(const cv::Mat& cvMat, const std::string& fullFilePath, const std::vector<int>& openCvCompressionParams = {CV_IMWRITE_JPEG_QUALITY, 100, CV_IMWRITE_PNG_COMPRESSION, 9});

    OPENPOSE_API cv::Mat loadImage(const std::string& fullFilePath, const int openCvFlags = CV_LOAD_IMAGE_ANYDEPTH);
}

#endif // OPENPOSE_FILESTREAM_FILE_STREAM_HPP
