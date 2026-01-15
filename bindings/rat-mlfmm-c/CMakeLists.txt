cmake_minimum_required(VERSION 3.20)
project(RatMlfmmC LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(RAT_MLFMM_DIR "" CACHE PATH "Path to rat-mlfmm source directory")
if(NOT RAT_MLFMM_DIR)
    message(FATAL_ERROR "RAT_MLFMM_DIR is required")
endif()
if(NOT EXISTS "${RAT_MLFMM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "RAT_MLFMM_DIR does not point to a rat-mlfmm source tree")
endif()

set(RAT_COMMON_DIR "" CACHE PATH "Path to rat-common source directory")
if(NOT RAT_COMMON_DIR)
    message(FATAL_ERROR "RAT_COMMON_DIR is required")
endif()
if(NOT EXISTS "${RAT_COMMON_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "RAT_COMMON_DIR does not point to a rat-common source tree")
endif()

set(JSONCPP_SRC_DIR "" CACHE PATH "Path to jsoncpp source directory")
if(NOT JSONCPP_SRC_DIR)
    message(FATAL_ERROR "JSONCPP_SRC_DIR is required")
endif()
if(NOT EXISTS "${JSONCPP_SRC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "JSONCPP_SRC_DIR does not point to a jsoncpp source tree")
endif()

set(ARMADILLO_SRC_DIR "" CACHE PATH "Path to armadillo source directory")
if(NOT ARMADILLO_SRC_DIR)
    message(FATAL_ERROR "ARMADILLO_SRC_DIR is required")
endif()
if(NOT EXISTS "${ARMADILLO_SRC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ARMADILLO_SRC_DIR does not point to an armadillo source tree")
endif()

set(TCLAP_SRC_DIR "" CACHE PATH "Path to tclap source directory")
if(NOT TCLAP_SRC_DIR)
    message(FATAL_ERROR "TCLAP_SRC_DIR is required")
endif()
if(NOT EXISTS "${TCLAP_SRC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "TCLAP_SRC_DIR does not point to a tclap source tree")
endif()

set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_WARNING_AS_ERROR OFF CACHE BOOL "" FORCE)
add_subdirectory(${JSONCPP_SRC_DIR} ${CMAKE_BINARY_DIR}/jsoncpp-build EXCLUDE_FROM_ALL)
set(JsonCPP_INCLUDE_DIR ${JSONCPP_SRC_DIR}/include CACHE PATH "" FORCE)
set(JsonCPP_LIBRARY ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}jsoncpp${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "" FORCE)

set(BUILD_SMOKE_TEST OFF CACHE BOOL "" FORCE)
set(ALLOW_BLAS_LAPACK_MACOS ON CACHE BOOL "" FORCE)
add_subdirectory(${ARMADILLO_SRC_DIR} ${CMAKE_BINARY_DIR}/armadillo-build EXCLUDE_FROM_ALL)
set(Armadillo_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE PATH "" FORCE)

set(TCLAP_INCLUDE_DIR ${TCLAP_SRC_DIR}/include CACHE PATH "" FORCE)

set(ENABLE_DOUBLE_PRECISION ON CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_BLAS ON CACHE BOOL "" FORCE)
set(ENABLE_SUPERLU OFF CACHE BOOL "" FORCE)
set(ENABLE_MKL_ALLOC OFF CACHE BOOL "" FORCE)

add_subdirectory(${RAT_COMMON_DIR} ${CMAKE_BINARY_DIR}/rat-common-build EXCLUDE_FROM_ALL)
set(RatCommon_DIR ${CMAKE_BINARY_DIR}/rat-common-build CACHE PATH "" FORCE)

set(ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_MATLAB OFF CACHE BOOL "" FORCE)

add_subdirectory(${RAT_MLFMM_DIR} ${CMAKE_BINARY_DIR}/rat-mlfmm-build EXCLUDE_FROM_ALL)

add_library(rat_mlfmm_c SHARED
    src/rat_mlfmm_c.cpp
    src/rat_mlfmm_c_err.cpp
)

target_compile_features(rat_mlfmm_c PRIVATE cxx_std_14)

target_include_directories(rat_mlfmm_c
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${RAT_MLFMM_DIR}/include
)

target_link_libraries(rat_mlfmm_c PRIVATE ratmlfmm Rat::Common)

set_target_properties(rat_mlfmm_c PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_compile_definitions(rat_mlfmm_c PRIVATE RAT_MLFMM_C_BUILD)
