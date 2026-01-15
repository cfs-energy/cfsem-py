# make a list of unit test files
set(example_list
	serialization.cpp
	tclapexample.cpp
)

# walk over source files
foreach(srcfile ${example_list})
	string(REPLACE ".cpp" "" name ${srcfile})
	add_executable(${name} ${srcfile})
	target_link_libraries(${name} Rat::Common)
endforeach()