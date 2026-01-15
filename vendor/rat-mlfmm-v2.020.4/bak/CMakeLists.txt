

# try to find the matlab library
find_package(Matlab)

# try to find the octave library
find_package(Octave)

# matlab was found
if(Matlab_FOUND)
    message(STATUS "MATLAB Found, MATLAB MEX will be compiled.")
    matlab_add_mex(NAME mexmlfmm SRC mexmlfmm.cpp LINK_TO rat-mlfmm-st ${Matlab_LIBRARIES})

elseif(Octave_FOUND)
    message(STATUS "Octave Found, Octave MEX will be compiled.")
    octave_add_mex(NAME mexmlfmm SRC mexmlfmm.cpp LINK_TO rat-mlfmm-st ${Matlab_LIBRARIES})

else()
    MESSAGE(STATUS "MATLAB not found, MATLAB MEX will not be compiled.")
endif()