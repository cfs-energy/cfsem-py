# compile the MLFMM library
file(GLOB SOURCE_FILES "*.cpp")

# this is the "object library" target: compiles the sources only once
# add_library(objlib OBJECT ${SOURCE_FILES})

# If we have compiler requirements for this library, list them here
# target_compile_features(objlib
#     PUBLIC cxx_auto_type
#     PRIVATE cxx_variadic_templates)

# shared and static libraries built from the same object files
# add_library(rat-common-st STATIC $<TARGET_OBJECTS:objlib>)
add_library(rat-commonlib SHARED 
	elements.cpp
	extra.cpp
	freecad.cpp
	gauss.cpp
	gmres.cpp
	gmshfile.cpp
	newtonraphson.cpp
	node.cpp
	opera.cpp
	prime.cpp
	serializer.cpp
	steepest.cpp)

# properties
set_target_properties(rat-commonlib PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(rat-commonlib PROPERTIES SOVERSION 1)

# add include directory
target_include_directories(rat-commonlib PRIVATE ../include)

# Depend on a library that we defined in the top-level file
# target_link_libraries(rat-common-st ${LIBS})
target_link_libraries(rat-commonlib ${LIBS})
