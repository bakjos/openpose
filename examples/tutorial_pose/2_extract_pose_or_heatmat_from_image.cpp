// ------------------------- OpenPose Library Tutorial - Pose - Example 2 - Extract Pose or Heatmap from Image -------------------------
// This second example shows the user how to:
    // 1. Load an image (`filestream` module)
    // 2. Extract the pose of that image (`pose` module)
    // 3. Render the pose or heatmap on a resized copy of the input image (`pose` module)
    // 4. Display the rendered pose or heatmap (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module: for the Array<float> class that the `pose` module needs
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             4,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased,"
                                                        " the speed increases.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you"
                                                        " want to change the initial scale, you actually want to multiply the `net_resolution` by"
                                                        " your desired initial scale.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              19,             "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");

op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (poseModeString == "COCO")
        return op::PoseModel::COCO_18;
    else if (poseModeString == "MPI")
        return op::PoseModel::MPI_15;
    else if (poseModeString == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        return op::PoseModel::COCO_18;
    }
}

// Google flags into program variables
std::tuple<op::Point<int>, op::Point<int>, op::Point<int>, op::PoseModel> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    op::Point<int> outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ",
               __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    op::Point<int> netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.x, &netInputSize.y);
    op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)",
               __LINE__, __FUNCTION__, __FILE__);
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
    // Logging and return result
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
}

int openPoseTutorialPose2()
{
    op::log("OpenPose Library Tutorial - Example 2.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
        // - 0 will output all the logging messages
        // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // Step 2 - Read Google flags (user defined configuration)
    op::Point<int> outputSize;
    op::Point<int> netInputSize;
    op::Point<int> netOutputSize;
    op::PoseModel poseModel;
    std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters();
    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    std::shared_ptr<op::PoseExtractor> poseExtractorPtr = std::make_shared<op::PoseExtractorCaffe>(netInputSize, netOutputSize, outputSize,
                                                                                                   FLAGS_num_scales, poseModel,
                                                                                                   FLAGS_model_folder, FLAGS_num_gpu_start);
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, poseExtractorPtr, !FLAGS_disable_blending, (float)FLAGS_alpha_pose,
                                 (float)FLAGS_alpha_heatmap};
    poseRenderer.setElementToRender(FLAGS_part_to_show);
    op::OpOutputToCvMat opOutputToCvMat{outputSize};
    const op::Point<int> windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 2"};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorPtr->initializationOnThread();
    poseRenderer.initializationOnThread();

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    cv::Mat inputImageCv = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
	cv::cuda::GpuMat inputImage;
	inputImage.upload(inputImageCv);
    if(inputImage.empty())
        op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Format input image to OpenPose input and output formats
    op::GpuArray<float> netInputArray;
    std::vector<float> scaleRatios;
    //std::tie(netInputArray, scaleRatios) = cvMatToOpInput.format(inputImage);
	scaleRatios = cvMatToOpInput.format(netInputArray, inputImage);
    float scaleInputToOutput;
    op::GpuArray<float> outputArray;
    //std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
    cvMatToOpOutput.format(inputImage, scaleInputToOutput, outputArray);
    // Step 3 - Estimate poseKeypoints
    poseExtractorPtr->forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorPtr->getPoseKeypoints();
    const auto scaleNetToOutput = poseExtractorPtr->getScaleNetToOutput();
    // Step 4 - Render pose
    poseRenderer.renderPose(outputArray, poseKeypoints, scaleNetToOutput);
    // Step 5 - OpenPose output format to cv::Mat
    //auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);
	cv::cuda::GpuMat outputImage;
	opOutputToCvMat.formatToCvMat(outputArray, outputImage);

	cv::Mat ouputCvImage;

	outputImage.download(ouputCvImage);
    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    frameDisplayer.displayFrame(ouputCvImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    // Step 2 - Logging information message
    op::log("Example 2 successfully finished.", op::Priority::High);
    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseTutorialPose2");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialPose2
    return openPoseTutorialPose2();
}
