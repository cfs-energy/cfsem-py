# make a list of unit test files
set(example_list
	scaling.cpp
	loop.cpp
	mlfmm_random.cpp
	element.cpp
	soleno.cpp
	# zonal.cpp
	integral_experiment.cpp
	charged.cpp
)

# add NVIDIA CUDA items
if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
	set(example_list ${example_list} managed_memory.cpp)
endif()

# walk over source files
foreach(srcfile ${example_list})
	string(REPLACE ".cpp" "" name ${srcfile})
	add_executable(${name} ${srcfile})
	target_link_libraries(${name} Rat::MLFMM)
	# if(CMAKE_CUDA_COMPILER)
	# 	set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CUDA)
	# endif()
endforeach()
