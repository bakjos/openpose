#ifndef OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
#define OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP

#include <atomic>
#include <mutex>
#include <sstream> // std::stringstream
#include <string>
#include <vector>
#include "enumClasses.hpp"
#include "macros.hpp"

namespace op
{
    template<typename T>
    std::string tToString(const T& message)
    {
        // Message -> ostringstream
        std::ostringstream oss;
        oss << message;
        // ostringstream -> std::string
        return oss.str();
    }

    // Error managment - How to use:
        // error(message, __LINE__, __FUNCTION__, __FILE__);
	OPENPOSE_API void error(const std::string& message, const int line = -1, const std::string& function = "", const std::string& file = "");

    template<typename T>
    inline void error(const T& message, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        error(tToString(message), line, function, file);
    }

    // Printing info - How to use:
        // log(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);  // It will print info if desiredPriority >= sPriorityThreshold
    OPENPOSE_API void log(const std::string& message, const Priority priority = Priority::Max, const int line = -1, const std::string& function = "", const std::string& file = "");

    template<typename T>
    inline void log(const T& message, const Priority priority = Priority::Max, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        log(tToString(message), priority, line, function, file);
    }

    // If only desired on debug mode (no computational cost at all on release mode):
        // dLog(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);  // It will print info if desiredPriority >= sPriorityThreshold
    template<typename T>
    inline void dLog(const T& message, const Priority priority = Priority::Max, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        #ifndef NDEBUG
            log(message, priority, line, function, file);
        #else
            UNUSED(message);
            UNUSED(priority);
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
        #endif
    }

    // This class is thread-safe
    class OPENPOSE_API ConfigureError
    {
    public:
        static std::vector<ErrorMode> getErrorModes();

        static void setErrorModes(const std::vector<ErrorMode>& errorModes);
    };

    // This class is thread-safe
    class OPENPOSE_API ConfigureLog
    {
    public:
        static Priority getPriorityThreshold();

        static const std::vector<LogMode>& getLogModes();

        static void setPriorityThreshold(const Priority priorityThreshold);

        static void setLogModes(const std::vector<LogMode>& loggingModes);
    };
}

#endif // OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
