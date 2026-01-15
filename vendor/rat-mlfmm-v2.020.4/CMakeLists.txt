## PROJECT SETUP
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(Rat-MLFMM VERSION 2.020.4 LANGUAGES CXX)

# set all policies to this version of cmake
cmake_policy(VERSION ${CMAKE_VERSION})

# switchboard
option(ENABLE_CUDA "build with CUDA GPU acceleration" ON)
option(ENABLE_CUDA_DOUBLE_PRECISION "build GPU code with double precision" OFF)
option(ENABLE_CUDA_FAST_MATH "use fast math for cuda routines" OFF)
option(ENABLE_TESTING "build unit/system tests" ON)
option(ENABLE_EXAMPLES "build examples" ON)
option(ENABLE_MATLAB "build matlab mex interface" OFF)

# where are the custom CMake modules located
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ensure out of core build
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
	message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n")
endif()

# set build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# compiler options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message(STATUS "using GCC settings")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3 -fgcse-after-reload")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	message(STATUS "using Clang settings")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3 -funroll-loops")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	message(STATUS "using MSVC settings")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
	message(STATUS "Compiler not recognized")
endif()

# windows specific
if(WIN32)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "" FORCE)

# report
message(STATUS "building in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# enable CUDA
if(ENABLE_CUDA)
	include("cmake/EnableLanguageCUDA.cmake")
	include("cmake/ConvertCUDAFlags.cmake")
else()
	message(STATUS "CUDA disabled.")
endif()

# EXTERNAL LIBRARIES
# find the rat common library
find_package(RatCommon 2.020.4 REQUIRED)
# if(RatCommon_FOUND)
# 	if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
# 		CUDA_CONVERT_FLAGS(Rat::Common)
# 	endif()
# endif()

## LIBRARY
# create a list of sources
set(source_list
	src/chargedpolyhedron.cpp
	src/background.cpp
	src/boundmesh.cpp
	src/charges.cpp
	src/currentmesh.cpp
	src/currentsources.cpp
	src/currentsurface.cpp
	src/extra.cpp
	src/fmmmat.cpp
	src/grid.cpp
	src/ilist.cpp
	src/interp.cpp
	src/mgntargets.cpp
	src/mlfmm.cpp
	src/momentsources.cpp
	src/multisources.cpp
	src/multitargets2.cpp
	src/multitargets.cpp
	src/nodelevel.cpp
	src/savart.cpp
	src/settings.cpp
	src/soleno.cpp
	src/sources.cpp
	src/spharm.cpp
	src/stars.cpp
	src/stmat.cpp
	src/targetpoints.cpp
	src/trackinggrid.cpp
	src/gpucurrentsources.cpp
	src/mgncharges.cpp
	src/mgnchargesurface.cpp
	src/sourcecoil.cpp
	src/brick8node.cpp
	src/solenosources.cpp
)

# add NVIDIA CUDA kernels to list
if(CMAKE_CUDA_COMPILER)
	set(source_list ${source_list} 
		src/gpukernels.cu)
endif()

# add the library
add_library(ratmlfmm SHARED ${source_list})

# Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(Rat::MLFMM ALIAS ratmlfmm)

# properties
set_target_properties(ratmlfmm PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(ratmlfmm PROPERTIES SOVERSION 1)

# add include directory
target_include_directories(ratmlfmm
	PUBLIC 
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)

# enable warnings
target_compile_options(ratmlfmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
	$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
		-Wall -Wextra -Wpedantic -Wsuggest-override -Wold-style-cast>
	$<$<CXX_COMPILER_ID:MSVC>:
		/W3 > >)

# link to external libraries
target_link_libraries(ratmlfmm PUBLIC Rat::Common)

# NVIDIA CUDA options
if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
	# declare CUDA KERNEL flag
	target_compile_definitions(ratmlfmm PUBLIC -DENABLE_CUDA_KERNELS)
	
	# precision
	if(ENABLE_CUDA_DOUBLE_PRECISION)
		target_compile_definitions(ratmlfmm PUBLIC -DRAT_CUDA_DOUBLE_PRECISION)
	endif()

	# set compilation
	set_target_properties(ratmlfmm PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

	# set libraries specific to cuda
	target_link_libraries(ratmlfmm PUBLIC CUDA::cublas)
	target_link_libraries(ratmlfmm PUBLIC CUDA::curand)

	# set linker language
	set_target_properties(ratmlfmm PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(ratmlfmm PROPERTIES CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120")	

	# use less accurate but faster math routines
	if(ENABLE_CUDA_FAST_MATH)
		target_compile_options(ratmlfmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
		target_compile_definitions(ratmlfmm PUBLIC -DENABLE_CUDA_FAST_MATH)
	endif()
endif()

# disable windows warnings
if(WIN32)
	target_compile_definitions(ratmlfmm PUBLIC -D_CRT_SECURE_NO_WARNINGS)
endif()

# Version header file
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_MAJOR_STRIPPED ${PROJECT_VERSION_MAJOR})
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_MINOR_STRIPPED ${PROJECT_VERSION_MINOR})
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_PATCH_STRIPPED ${PROJECT_VERSION_PATCH})
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hh.in 
	${CMAKE_BINARY_DIR}/include/version.hh)

# build unit tests
if(ENABLE_TESTING)
	# allow for unit tests
	enable_testing()
	
	# add test directory
	add_subdirectory(test)
endif(ENABLE_TESTING)

# examples
if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
endif(ENABLE_EXAMPLES)

# find MATLAB for compiling mex functions
# make sure it also finds the mxlibrary component
if(ENABLE_MATLAB)
	add_subdirectory(matlab)
endif(ENABLE_MATLAB)


## INSTALLATION
# get install directories
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ratmlfmm)

install(TARGETS ratmlfmm
	EXPORT ratmlfmm-targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rat
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/rat
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# This is required so that the exported target has the name RatCmn and not ratmlfmm
set_target_properties(ratmlfmm PROPERTIES EXPORT_NAME MLFMM)

# Install header files
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rat/mlfmm)
install(FILES ${CMAKE_BINARY_DIR}/include/version.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rat/mlfmm)

# Export the targets to a script
install(EXPORT ratmlfmm-targets
	FILE
		RatMLFMMTargets.cmake
	NAMESPACE
		Rat::
	DESTINATION
		${INSTALL_CONFIGDIR}
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/RatMLFMMConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/RatMLFMMConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/RatMLFMMConfig.cmake
	INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install the config, configversion and custom find modules
install(FILES
	${CMAKE_CURRENT_LIST_DIR}/cmake/ConvertCUDAFlags.cmake
	${CMAKE_CURRENT_LIST_DIR}/cmake/EnableLanguageCUDA.cmake
	${CMAKE_CURRENT_BINARY_DIR}/RatMLFMMConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/RatMLFMMConfigVersion.cmake
	DESTINATION ${INSTALL_CONFIGDIR}
)

## EXPORT
# Exporting from the build tree
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/ConvertCUDAFlags.cmake
	${CMAKE_CURRENT_BINARY_DIR}/ConvertCUDAFlags.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/EnableLanguageCUDA.cmake
	${CMAKE_CURRENT_BINARY_DIR}/EnableLanguageCUDA.cmake COPYONLY)

export(EXPORT ratmlfmm-targets
	FILE ${CMAKE_CURRENT_BINARY_DIR}/RatMLFMMTargets.cmake
	NAMESPACE Rat::)

# Register package in user's package registry
export(PACKAGE RatMLFMM)

#####################
# Setup install paths
#####################

include(GNUInstallDirs)

# Set RPATHS
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

# Create Debian package, must be the last thing
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(GenerateDebianPackage)
