
# find octave
find_package(OCTAVE QUIET)

# create the octave mex file
if(OCTAVE_FOUND)
    add_library(mexmlfmm SHARED mexmlfmm.cpp )
    target_link_libraries (mexmlfmm Octave::Octave)
    target_link_libraries (mexmlfmm Rat::MLFMM)
    set_target_properties (mexmlfmm PROPERTIES PREFIX "" SUFFIX  ".mex")
    set_property(TARGET ${FUNCTIONNAME} PROPERTY CXX_STANDARD 11)
endif()

# find matlab
find_package(Matlab QUIET)

# create the matlab mex file
if(Matlab_FOUND)
    matlab_add_mex(NAME mexmlfmm SRC mexmlfmm.cpp)
    target_link_libraries (mexmlfmm Rat::MLFMM)
endif()