# - Try to find CAFFE
#
# The following variables are optionally searched for defaults
#  CAFFE_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  CAFFE_FOUND
#  CAFFE_INCLUDE_DIRS
#  CAFFE_LIBRARIES

include(FindPackageHandleStandardArgs)

set(CAFFE_ROOT_DIR "" CACHE PATH "Folder containing CAFFE")

find_path(CAFFE_INCLUDE_DIR caffe/caffe.hpp
  PATHS ${CAFFE_ROOT_DIR}
  PATH_SUFFIXES
  include)

if(MSVC)
    find_library(CAFFE_LIBRARY_RELEASE caffe
        PATHS ${CAFFE_ROOT_DIR}
        PATH_SUFFIXES lib/Release build/lib build/lib/release)

    find_library(CAFFE_LIBRARY_DEBUG caffe-d
        PATHS ${CAFFE_ROOT_DIR}
        PATH_SUFFIXES lib/Debug build/lib build/lib/debug)

    set(CAFFE_LIBRARY optimized ${CAFFE_LIBRARY_RELEASE} debug ${CAFFE_LIBRARY_DEBUG})

    find_library(CAFFE_PROTO_LIBRARY_RELEASE proto
        PATHS ${CAFFE_ROOT_DIR}
        PATH_SUFFIXES lib/Release build/lib build/lib/release)

    find_library(CAFFE_PROTO_LIBRARY_DEBUG proto-d
        PATHS ${CAFFE_ROOT_DIR}
        PATH_SUFFIXES lib/Debug build/lib build/lib/debug)

    set(CAFFE_PROTO_LIBRARY optimized ${CAFFE_PROTO_LIBRARY_RELEASE} debug ${CAFFE_PROTO_LIBRARY_DEBUG})


else()
    find_library(CAFFE_LIBRARY caffe
  PATHS ${CAFFE_ROOT_DIR}
  PATH_SUFFIXES
  build/lib)
endif()





find_package_handle_standard_args(CAFFE DEFAULT_MSG
  CAFFE_INCLUDE_DIR CAFFE_LIBRARY)

if(CAFFE_FOUND)
  set(CAFFE_INCLUDE_DIRS ${CAFFE_INCLUDE_DIR})
  set(CAFFE_LIBRARIES ${CAFFE_LIBRARY})
  if ( MSVC ) 
  	list(APPEND CAFFE_LIBRARIES ${CAFFE_PROTO_LIBRARY})
  endif()
 
endif()