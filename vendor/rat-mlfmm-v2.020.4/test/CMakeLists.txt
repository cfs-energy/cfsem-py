# make a list of unit test files
set(test_list
	test_factorial.cpp
	test_nm2fidx.cpp
	test_solenoid.cpp
	test_interp.cpp
	test_mlfmm_random.cpp
	test_ring_axial.cpp
	test_solenoid_mesh.cpp
	test_matrices.cpp
	test_morton.cpp
	test_ring_radial.cpp
	test_spharm.cpp
	test_mp2lp_direct.cpp
	test_shell.cpp
	test_stellar.cpp
	test_mlfmm_loop.cpp
	test_multi.cpp
	test_soleno.cpp
	test_tracking.cpp
	test_mgncharges.cpp
	test_twocharges.cpp
	test_charged_triangle.cpp
	test_charged_quadrilateral.cpp
	test_surface_integrals.cpp
	test_source_coil.cpp
)

# walk over source files
foreach(srcfile ${test_list})
	string(REPLACE ".cpp" "" name ${srcfile})
	set(name ${name})
	add_executable(${name} ${srcfile})
	target_link_libraries(${name} Rat::MLFMM)
	add_test(${name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})
endforeach()

# add NVIDIA CUDA tests to list
if(CMAKE_CUDA_COMPILER)
	set(gpu_test_list
		test_mlfmm_random.cpp
		test_mlfmm_loop.cpp)

	# walk over source files
	foreach(srcfile ${gpu_test_list})
		string(REPLACE ".cpp" "_gpu" name ${srcfile})
		add_executable(${name} ${srcfile})
		target_link_libraries(${name} Rat::MLFMM)
		add_test(${name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})
		target_compile_definitions(${name} PRIVATE ENABLE_GPU)
	endforeach()
endif()


