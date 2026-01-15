## PROJECT SETUP
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(Rat-Common VERSION 2.020.4 LANGUAGES CXX C)

# set all policies to this version of cmake
cmake_policy(VERSION ${CMAKE_VERSION})

# switchboard
option(ENABLE_DOUBLE_PRECISION "build with double precision floats" ON) # enable here for all rat libraries
option(ENABLE_TESTING "build unit/system tests" ON)
option(ENABLE_EXAMPLES "build examples" ON)
option(ENABLE_BLAS "build with optimized BLAS library" ON)
option(ENABLE_SUPERLU "link with superlu" ON)
option(ENABLE_MKL_ALLOC "enable memory allocation with mkl" OFF)

# compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
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

## EXTERNAL LIBRARIES
# where are the custom CMake modules located
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find posix threads package
if(APPLE)
set(CMAKE_THREAD_LIBS_INIT "-lpthread") # needed for osx?
endif()
find_package(Threads REQUIRED)

# find JSON CPP library and create interface
find_package(JsonCPP REQUIRED)

# tclap command line input parser
find_package(TCLAP REQUIRED)

# find armadillo 
find_package(Armadillo 12.8.0 CONFIG REQUIRED)

# On some systems superlu needs to be explcitly linked against
if(ENABLE_SUPERLU)
		if(Armadillo_VERSION VERSION_GREATER_EQUAL 14.4.0)
				find_package(superlu)
		else()
				find_package(superlu EXACT 5.3.0) # before armadillo 14.4.0
		endif()
endif()

# check if optimized BLAS enabled
if(ENABLE_BLAS)
	find_package(LAPACK REQUIRED)
	if (NOT BLAS_FOUND)
		find_package(BLAS REQUIRED)
	endif()

	# create an interface for BLAS
	if(BLAS_FOUND)
		# message(STATUS "using blas: ${BLAS_LIBRARIES}")
		if(NOT TARGET BLAS::BLAS)
			add_library(BLAS::BLAS INTERFACE IMPORTED)
			set_target_properties(BLAS::BLAS PROPERTIES
				INTERFACE "${BLAS_LIBRARIES}")
		endif()
	endif()

	# create an interface for LAPACK
	if(LAPACK_FOUND)
		if(NOT TARGET LAPACK::LAPACK)
			add_library(LAPACK::LAPACK INTERFACE IMPORTED)
			set_target_properties(LAPACK::LAPACK PROPERTIES
				INTERFACE "${LAPACK_LIBRARIES}")
		endif()
	endif()
endif()

# find armadillo library
if(NOT TARGET Arma::Armadillo)
	# create target
	add_library(Arma::Armadillo INTERFACE IMPORTED)
	set_target_properties(Arma::Armadillo PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
		INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}")

	# disable opernmp acceleration
	# target_compile_definitions(Arma::Armadillo INTERFACE -DARMA_DONT_USE_OPENMP)
	# target_compile_definitions(Arma::Armadillo INTERFACE -DARMA_DONT_USE_HDF5)

	# set (n)debug flags for Armadilo
	if(CMAKE_BUILD_TYPE STREQUAL Release)
		message(STATUS "Using Armadillo no-debug flags (conformance check disabled!)")
		target_compile_definitions(Arma::Armadillo INTERFACE -DNDEBUG -DARMA_NO_DEBUG -DARMA_DONT_CHECK_CONFORMANCE -DARMA_WARN_LEVEL=1)
	else()
		message(STATUS "Using Armadillo in debug mode")
	endif()

	# disable armadillo wrapper (for using custom BLAS)
	if(BLAS_FOUND AND LAPACK_FOUND)
		# report to user
		message(STATUS "Disabling Armadillo blas wrapper")
		target_compile_definitions(Arma::Armadillo INTERFACE -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS)
	endif()

	# disable superlu if not available
	if(superlu_FOUND)
		message(STATUS "Enabling Superlu")
		target_compile_definitions(Arma::Armadillo INTERFACE -DARMA_USE_SUPERLU)
	else()
		message(STATUS "Disabling Armadillo SuperLU")
		target_compile_definitions(Arma::Armadillo INTERFACE -DARMA_DONT_USE_SUPERLU)
	endif()
endif()


## LIBRARY
# create a list of sources
# ls *.cpp >> CMakeLists.txt
set(source_list
	src/abaqusfile.cpp
	src/gauss.cpp
	src/newtonraphson.cpp
	src/prime.cpp
	src/stlfile.cpp
	src/elements.cpp
	src/gmres.cpp
	src/node.cpp
	src/rungekutta.cpp
	src/extra.cpp
	src/gmshfile.cpp
	src/nodegroup.cpp
	src/serializer.cpp
	src/freecad.cpp
	src/log.cpp
	src/opera.cpp
	src/steepest.cpp
	src/marchingcubes.cpp
	src/parfor.cpp
)

# add the common library
add_library(ratcmn SHARED ${source_list})

# Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(Rat::Common ALIAS ratcmn)

# properties
set_target_properties(ratcmn PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(ratcmn PROPERTIES SOVERSION 1)

# add include directory
target_include_directories(ratcmn
	PUBLIC 
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)

# disable windows warnings
if(WIN32)
	target_compile_definitions(ratcmn PUBLIC -D_CRT_SECURE_NO_WARNINGS)
endif()

# enable warnings
target_compile_options(ratcmn PRIVATE 
	$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
		-Wpedantic -Wall -Wextra -Wsuggest-override -Wold-style-cast>
	$<$<CXX_COMPILER_ID:MSVC>:
		/W3>)

# disable multi-threading in armadillo 
target_compile_definitions(ratcmn PUBLIC -DARMA_DONT_USE_HDF5)
# target_compile_definitions(ratcmn PUBLIC -DARMA_DONT_USE_OPENMP)

# enable intel-mkl num threads
if(ENABLE_MKL_ALLOC)
	target_include_directories(ratcmn PUBLIC /opt/intel/mkl/include)
	target_compile_definitions(ratcmn PUBLIC -DARMA_USE_MKL_ALLOC)
endif()

# link libraries
target_link_libraries(ratcmn PUBLIC
		Arma::Armadillo
		JsonCPP::JsonCPP
		Threads::Threads)

# link superlu if enabled and found
if(ENABLE_SUPERLU AND superlu_FOUND)
	message(STATUS "Linking with superlu")
	target_link_libraries(ratcmn PUBLIC superlu) # interfaces applied here
endif()

# floating point precision
if(ENABLE_DOUBLE_PRECISION)
	message(STATUS "Building with double precision floats")
	target_compile_definitions(ratcmn PUBLIC -DRAT_DOUBLE_PRECISION)
else()
	message(STATUS "Building with single precision floats")
endif()

# filesystem library
set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost 1.78.0 REQUIRED CONFIG COMPONENTS filesystem iostreams thread chrono) # recently removed system component here
if(Boost_FOUND)
	target_compile_definitions(ratcmn PUBLIC -DBOOST_FILESYSTEM_NO_DEPRECATED)
	target_link_libraries(ratcmn PUBLIC Boost::filesystem Boost::iostreams Boost::boost Boost::thread Boost::chrono)
endif()

# set Armadillo BLAS wrapper
if(ENABLE_BLAS)
	# disable armadillo wrapper (for using custom BLAS)
	if(BLAS_FOUND AND LAPACK_FOUND)
		# link to blas libraries
		target_link_libraries(ratcmn PUBLIC BLAS::BLAS)
		target_link_libraries(ratcmn PUBLIC LAPACK::LAPACK)
	else()
		# report to user
		message(STATUS "Using armadillo blas wrapper.")
	endif()
endif()

# Version header file
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_MAJOR_STRIPPED ${PROJECT_VERSION_MAJOR})
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_MINOR_STRIPPED ${PROJECT_VERSION_MINOR})
string(REGEX REPLACE "^0+" "" PROJECT_VERSION_PATCH_STRIPPED ${PROJECT_VERSION_PATCH})
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hh.in 
	${CMAKE_CURRENT_BINARY_DIR}/include/version.hh)

## UNIT/SYSTEM TESTS
# check switchboard
if(ENABLE_TESTING)
	# use cmake's platform for unit testing
	enable_testing()

	# add testing directory
	add_subdirectory(test)
endif()


## EXAMPLES
# check switchboard
if(ENABLE_EXAMPLES)
	# add example directory
	add_subdirectory(examples)
endif()


## INSTALLATION
# get install directories
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ratcommon)

install(TARGETS ratcmn
	EXPORT ratcmn-targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rat
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/rat
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# This is required so that the exported target has the name RatCmn and not ratcmn
set_target_properties(ratcmn PROPERTIES EXPORT_NAME Common)

# install includes
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rat/common)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/version.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rat/common)

# Export the targets to a script
install(EXPORT ratcmn-targets
	FILE RatCommonTargets.cmake
	NAMESPACE Rat::
	DESTINATION ${INSTALL_CONFIGDIR}
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/RatCommonConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
	${CMAKE_CURRENT_LIST_DIR}/cmake/RatCommonConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/RatCommonConfig.cmake
	INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
	PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

# Install the config, configversion and custom find modules
install(FILES
	${CMAKE_CURRENT_LIST_DIR}/cmake/FindJsonCPP.cmake
	${CMAKE_CURRENT_LIST_DIR}/cmake/FindTCLAP.cmake
	${CMAKE_CURRENT_LIST_DIR}/cmake/Findsuperlu.cmake
	${CMAKE_CURRENT_BINARY_DIR}/RatCommonConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/RatCommonConfigVersion.cmake
	DESTINATION ${INSTALL_CONFIGDIR}
)

## EXPORT
# Exporting from the build tree
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindJsonCPP.cmake
	${CMAKE_CURRENT_BINARY_DIR}/FindJsonCPP.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindTCLAP.cmake
	${CMAKE_CURRENT_BINARY_DIR}/FindTCLAP.cmake COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/Findsuperlu.cmake
	${CMAKE_CURRENT_BINARY_DIR}/Findsuperlu.cmake COPYONLY)

export(EXPORT ratcmn-targets
	FILE ${CMAKE_CURRENT_BINARY_DIR}/RatCommonTargets.cmake
	NAMESPACE Rat::)

# Register package in user's package registry
export(PACKAGE RatCommon)


#####################
# Setup install paths
#####################

include(GNUInstallDirs)

# Set RPATHS
# SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

# Create Debian package, must be the last thing
include(GenerateDebianPackage)


