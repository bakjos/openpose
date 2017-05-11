cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

project(OpenPose C CXX)

set(OPENPOSE_TARGET_VERSION "1.0.0-rc1" CACHE STRING "OpenPose logical version")
set(OPENPOSE_TARGET_SOVERSION "1.0.0-rc1" CACHE STRING "OpenPose soname version")
add_definitions(-DCAFFE_VERSION=${OPENPOSE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)


include(ExternalProject)
include(GNUInstallDirs)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/TargetResolvePrerequesites.cmake)



openpose_option(USE_BOOST_DYNAMIC "Use the dynamic libraries " ON IF MSVC)

openpose_option(USE_CUDNN "Build OpenPose with cuDNN library support" ON IF NOT CPU_ONLY)
openpose_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
openpose_option(COPY_PREREQUISITES "Copy the prerequisites next to each executable or shared library directory" ON IF MSVC)
openpose_option(INSTALL_PREREQUISITES "Install the prerequisites next to each executable or shared library directory" ON IF MSVC)
openpose_option(USE_LEVELDB "Build with levelDB" ON)
openpose_option(USE_LMDB "Build with lmdb" ON)
openpose_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)



set(OpenPose_LINKER_LIBS "")
set(OpenPose_INCLUDE_DIRS "")
set(OpenPose_DEFINITIONS "")
set(OpenPose_COMPILE_OPTIONS "")
set(OpenPose_DEBUG_POSTFIX "-d")

openpose_option(protobuf_MODULE_COMPATIBLE "Make the protobuf-config.cmake compatible with the module mode" ON IF MSVC)

include(cmake/WindowsDownloadPrebuiltDependencies.cmake)



 list(APPEND OpenPose_DEFINITIONS PUBLIC -DUSE_CAFFE)

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
    list(APPEND OpenPose_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
    list(APPEND OpenPose_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
    list(APPEND OpenPose_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
    list(APPEND OpenPose_DEFINITIONS PUBLIC -DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${vecLib_INCLUDE_DIR})
  list(APPEND OpenPose_LINKER_LIBS PUBLIC ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND OpenPose_DEFINITIONS PUBLIC -DUSE_ACCELERATE)
    endif()
  endif()
endif()




# ---[ Includes
set(OpenPose_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(OpenPose_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})


# ---[ CAFFE 
find_package(CAFFE REQUIRED)
list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${CAFFE_INCLUDE_DIRS})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${CAFFE_LIBRARIES})

# ---[ Boost
find_package(Boost 1.55 REQUIRED COMPONENTS system filesystem thread)

list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
list(APPEND OpenPose_DEFINITIONS PUBLIC -DBOOST_ALL_NO_LIB)
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

if ( USE_BOOST_DYNAMIC ) 
	list(APPEND OpenPose_DEFINITIONS PUBLIC -DBOOST_ALL_DYN_LINK)
endif()


# ---[ Google-glog
include("cmake/External/glog.cmake")
list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
find_package( Protobuf REQUIRED )
list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${PROTOBUF_INCLUDE_DIR})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${PROTOBUF_LIBRARIES})

if(MSVC)
  # Find HDF5 using it's hdf5-config.cmake file with MSVC
  if(DEFINED HDF5_DIR)
    list(APPEND CMAKE_MODULE_PATH ${HDF5_DIR})
  endif()
  find_package(HDF5 COMPONENTS C HL REQUIRED)
  set(HDF5_LIBRARIES hdf5-shared)
  set(HDF5_HL_LIBRARIES hdf5_hl-shared)
else()
  find_package(HDF5 COMPONENTS HL REQUIRED)
endif()
list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})


# ---[ OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)

if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
	find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
list(APPEND OpenPose_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
list(APPEND OpenPose_DEFINITIONS PUBLIC -DUSE_OPENCV)

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
  list(APPEND OpenPose_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    list(APPEND OpenPose_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  list(APPEND OpenPose_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
  list(APPEND OpenPose_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
  list(APPEND OpenPose_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif()


if ( WIN32) 
	list(APPEND OpenPose_DEFINITIONS PUBLIC -DCMAKE_WINDOWS_BUILD)
endif()



include(cmake/Cuda.cmake)

openpose_set_openpose_link()

# pass include pathes to cuda_include_directories()
set(OpenPose_ALL_INCLUDE_DIRS ${OpenPose_INCLUDE_DIRS})
list(REMOVE_ITEM OpenPose_ALL_INCLUDE_DIRS PRIVATE PUBLIC)

cuda_include_directories(${OpenPose_INCLUDE_DIR} ${OpenPose_SRC_DIR} ${OpenPose_ALL_INCLUDE_DIRS})

# add definitions to nvcc flags directly
set(OpenPose_ALL_DEFINITIONS ${OpenPose_DEFINITIONS})
list(REMOVE_ITEM OpenPose_ALL_DEFINITIONS PRIVATE PUBLIC)
list(APPEND CUDA_NVCC_FLAGS ${OpenPose_ALL_DEFINITIONS})




add_subdirectory(src/openpose)
add_subdirectory(examples)

