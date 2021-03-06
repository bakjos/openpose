file(GLOB examples_fld "${PROJECT_SOURCE_DIR}/examples/*")


foreach(source_folder ${examples_fld})
  file(GLOB_RECURSE example_srcs "${source_folder}/*.cpp")
  get_filename_component(source_name ${source_folder} NAME_WE)
  if ( example_srcs)

  	foreach(source_file ${example_srcs})
  		get_filename_component(name ${source_file} NAME_WE)
      set(name ${name}_example)
		  add_executable(${name} ${source_file})
      
		  target_compile_definitions(${name} ${OpenPose_DEFINITIONS})
		  target_link_libraries(${name} ${OpenPose_LINK})
	    openpose_default_properties(${name})	    

      if(MSVC AND COPY_PREREQUISITES)
        openpose_copy_prerequisites(${name} USE_HARD_LINKS)
      endif()

      if ( MSVC )

        set_target_properties(${name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        set_target_properties(${name} PROPERTIES FOLDER examples/${source_name})
      endif()

      # install
      install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})


      if(MSVC AND INSTALL_PREREQUISITES)
        openpose_install_prerequisites(${name} DESTINATION bin)
      endif()

  	endforeach()
  	
  endif()
endforeach()