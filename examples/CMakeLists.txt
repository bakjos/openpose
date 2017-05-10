file(GLOB examples_fld "${PROJECT_SOURCE_DIR}/examples/*")

foreach(source_folder ${examples_fld})
  file(GLOB_RECURSE example_srcs "${source_folder}/*.cpp")

  if ( example_srcs)

  	foreach(source_file ${example_srcs})
  		get_filename_component(name ${source_file} NAME_WE)
	  	message("name: " ${name})  

		add_executable(${name} ${source_file})
		target_compile_definitions(${name} ${OpenPoseExample_DEFINITIONS})
		target_link_libraries(${name} ${OpenPose_LINK})
	    openpose_default_properties(${name})	    
  	endforeach()
  	
  endif()
endforeach()