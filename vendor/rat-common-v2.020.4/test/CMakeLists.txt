# make a list of unit test files
set(test_list 
	test_bitshift.cpp
	test_hex_isinside.cpp
	test_quad_coords.cpp
	test_conncomp.cpp
	test_hex_jacobian1.cpp
	test_quad_jacobian.cpp
	test_cross.cpp
	test_hex_jacobian2.cpp
	test_rotation.cpp
	test_field2mat.cpp
	test_hex_volume.cpp
	test_runge_kutta.cpp
	test_find_sections.cpp
	test_ismember.cpp
	test_steepest.cpp
	test_find_unique.cpp
	test_line_jacobian.cpp
	test_tet_jacobian.cpp
	test_gauss2nodes_hex.cpp
	test_gauss2nodes_tet.cpp
	test_interp2.cpp
	test_interp3.cpp
	test_lininterp1.cpp
	test_lininterp2.cpp
	test_lininterp3.cpp
	test_triangle_jacobian.cpp
	test_gmres.cpp
	test_newton.cpp
	test_hex_coords.cpp
	test_parsort.cpp
	test_normalize.cpp
	test_jsonnan.cpp
	test_parfor.cpp
)

# walk over source files
foreach(srcfile ${test_list})
	get_filename_component(name ${srcfile} NAME_WE)
	add_executable(${name} ${srcfile})
	target_link_libraries(${name} Rat::Common)
	add_test(${name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})
endforeach()
