################################################################################################
# Defines global Caffe_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(openpose_set_openpose_link)
  if(BUILD_SHARED_LIBS)
    set(OpenPose_LINK openpose)
    set(OpenPoseExample_DEFINITIONS PUBLIC -DOPENPOSE_IMPORT_DLL)
  else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(OpenPose_LINK -Wl,-force_load openpose)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(OpenPose_LINK -Wl,--whole-archive openpose -Wl,--no-whole-archive)
    elseif(MSVC)
      set(OpenPose_LINK openpose)
    endif()
    set(OpenPoseExample_DEFINITIONS "")
  endif()
endmacro()
################################################################################################
# Convenient command to setup source group for IDEs that support this feature (VS, XCode)
# Usage:
#   openpose_source_group(<group> GLOB[_RECURSE] <globbing_expression>)
function(openpose_source_group group)
  cmake_parse_arguments(CAFFE_SOURCE_GROUP "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(CAFFE_SOURCE_GROUP_GLOB)
    file(GLOB srcs1 ${CAFFE_SOURCE_GROUP_GLOB})
    source_group(${group} FILES ${srcs1})
  endif()

  if(CAFFE_SOURCE_GROUP_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${CAFFE_SOURCE_GROUP_GLOB_RECURSE})
    source_group(${group} FILES ${srcs2})
  endif()
endfunction()

################################################################################################
# Collecting sources from globbing and appending to output list variable
# Usage:
#   openpose_collect_sources(<output_variable> GLOB[_RECURSE] <globbing_expression>)
function(openpose_collect_sources variable)
  cmake_parse_arguments(CAFFE_COLLECT_SOURCES "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(CAFFE_COLLECT_SOURCES_GLOB)
    file(GLOB srcs1 ${CAFFE_COLLECT_SOURCES_GLOB})
    set(${variable} ${variable} ${srcs1})
  endif()

  if(CAFFE_COLLECT_SOURCES_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${CAFFE_COLLECT_SOURCES_GLOB_RECURSE})
    set(${variable} ${variable} ${srcs2})
  endif()
endfunction()

################################################################################################
# Short command getting openpose sources (assuming standard Caffe code tree)
# Usage:
#   openpose_pickup_openpose_sources(<root>)
function(openpose_pickup_openpose_sources root)
  # put all files in source groups (visible as subfolder in many IDEs)
  openpose_source_group("Include"        GLOB "${root}/include/openpose/*.hpp*")
  openpose_source_group("Include\\core"  GLOB "${root}/include/openpose/core/*.hpp*")
  openpose_source_group("Include\\gui"  GLOB "${root}/include/openpose/gui/*.hpp*")
  openpose_source_group("Include\\pose"  GLOB "${root}/include/openpose/pose/*.hpp*")
  openpose_source_group("Include\\filestream" GLOB "${root}/include/openpose/filestream/*.hpp")
  openpose_source_group("Include\\producer" GLOB "${root}/include/openpose/producer/*.hpp")
  openpose_source_group("Include\\thread" GLOB "${root}/include/openpose/thread/*.hpp")
  openpose_source_group("Include\\utilities" GLOB "${root}/include/openpose/utilities/*.hpp")
  openpose_source_group("Include\\wrapper" GLOB "${root}/include/openpose/wrapper/*.hpp")
  openpose_source_group("Include\\experimental" GLOB "${root}/include/openpose/experimental/*.hpp")
  openpose_source_group("Include\\experimental\\face" GLOB "${root}/include/openpose/experimental/face/*.hpp")
  openpose_source_group("Include\\experimental\\filestream" GLOB "${root}/include/openpose/experimental/filestream/*.hpp")
  openpose_source_group("Include\\experimental\\hand" GLOB "${root}/include/openpose/experimental/hand/*.hpp")
  
  openpose_source_group("Source\\core"   GLOB "${root}/src/openpose/core/*.cpp")
  openpose_source_group("Source\\gui"   GLOB "${root}/src/openpose/gui/*.cpp")
  openpose_source_group("Source\\pose" GLOB "${root}/src/openpose/pose/*.cpp")
  openpose_source_group("Source\\filestream" GLOB "${root}/src/openpose/filestream/*.cpp")
  openpose_source_group("Source\\producer" GLOB "${root}/src/openpose/producer/*.cpp")
  openpose_source_group("Source\\thread" GLOB "${root}/src/openpose/thread/*.cpp")
  openpose_source_group("Source\\utilities" GLOB "${root}/src/openpose/utilities/*.cpp")
  openpose_source_group("Source\\wrapper" GLOB "${root}/src/openpose/wrapper/*.cpp")
  openpose_source_group("Source\\experimental" GLOB "${root}/src/openpose/experimental/face/*.cpp")
  openpose_source_group("Source\\experimental" GLOB "${root}/src/openpose/experimental/filestream/*.cpp")
  openpose_source_group("Source\\experimental" GLOB "${root}/src/openpose/experimental/hand/*.cpp")

  openpose_source_group("Source\\Cuda"   GLOB "${root}/src/openpose/pose/*.cu")
  openpose_source_group("Source\\Cuda"   GLOB "${root}/src/openpose/util/*.cu")
  openpose_source_group("Source\\Cuda"   GLOB "${root}/src/openpose/core/*.cu")
  openpose_source_group("Source\\Cuda"   GLOB "${root}/src/openpose/experimental/face/*.cu")
  openpose_source_group("Source\\Cuda"   GLOB "${root}/src/openpose/experimental/hand/*.cu")


  # source groups for test target
  #openpose_source_group("Include"      GLOB "${root}/include/openpose/test/test_*.h*")
  #openpose_source_group("Source"       GLOB "${root}/src/openpose/test/test_*.cpp")
  #openpose_source_group("Source\\Cuda" GLOB "${root}/src/openpose/test/test_*.cu")

  # collect files
  #file(GLOB test_hdrs    ${root}/include/openpose/test/test_*.h*)
  #file(GLOB test_srcs    ${root}/src/openpose/test/test_*.cpp)
  file(GLOB_RECURSE hdrs ${root}/include/openpose/*.h*)
  file(GLOB_RECURSE srcs ${root}/src/openpose/*.cpp)
  #list(REMOVE_ITEM  hdrs ${test_hdrs})
  #list(REMOVE_ITEM  srcs ${test_srcs})

  # adding headers to make the visible in some IDEs (Qt, VS, Xcode)
  list(APPEND srcs ${hdrs})
  #list(APPEND test_srcs ${test_hdrs})

  # collect cuda files
  #file(GLOB    test_cuda ${root}/src/openpose/test/test_*.cu)
  file(GLOB_RECURSE cuda ${root}/src/openpose/*.cu)
  #list(REMOVE_ITEM  cuda ${test_cuda})

  # add proto to make them editable in IDEs too
  #file(GLOB_RECURSE proto_files ${root}/src/openpose/*.proto)
  #list(APPEND srcs ${proto_files})

  # convert to absolute paths
  openpose_convert_absolute_paths(srcs)
  openpose_convert_absolute_paths(cuda)
  #openpose_convert_absolute_paths(test_srcs)
  #openpose_convert_absolute_paths(test_cuda)

  # propagate to parent scope
  set(srcs ${srcs} PARENT_SCOPE)
  set(cuda ${cuda} PARENT_SCOPE)
  #set(test_srcs ${test_srcs} PARENT_SCOPE)
  #set(test_cuda ${test_cuda} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for setting default target properties
# Usage:
#   openpose_default_properties(<target>)
function(openpose_default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX ${OpenPose_DEBUG_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  # make sure we build all external dependencies first
  if (DEFINED external_project_dependencies)
    add_dependencies(${target} ${external_project_dependencies})
  endif()
endfunction()

################################################################################################
# Short command for setting runtime directory for build target
# Usage:
#   openpose_set_runtime_directory(<target> <dir>)
function(openpose_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()

################################################################################################
# Short command for setting solution folder property for target
# Usage:
#   openpose_set_solution_folder(<target> <folder>)
function(openpose_set_solution_folder target folder)
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
  endif()
endfunction()

################################################################################################
# Reads lines from input file, prepends source directory to each line and writes to output file
# Usage:
#   openpose_configure_testdatafile(<testdatafile>)
function(openpose_configure_testdatafile file)
  file(STRINGS ${file} __lines)
  set(result "")
  foreach(line ${__lines})
    set(result "${result}${PROJECT_SOURCE_DIR}/${line}\n")
  endforeach()
  file(WRITE ${file}.gen.cmake ${result})
endfunction()

################################################################################################
# Filter out all files that are not included in selected list
# Usage:
#   openpose_leave_only_selected_tests(<filelist_variable> <selected_list>)
function(openpose_leave_only_selected_tests file_list)
  if(NOT ARGN)
    return() # blank list means leave all
  endif()
  string(REPLACE "," ";" __selected ${ARGN})
  list(APPEND __selected openpose_main)

  set(result "")
  foreach(f ${${file_list}})
    get_filename_component(name ${f} NAME_WE)
    string(REGEX REPLACE "^test_" "" name ${name})
    list(FIND __selected ${name} __index)
    if(NOT __index EQUAL -1)
      list(APPEND result ${f})
    endif()
  endforeach()
  set(${file_list} ${result} PARENT_SCOPE)
endfunction()

