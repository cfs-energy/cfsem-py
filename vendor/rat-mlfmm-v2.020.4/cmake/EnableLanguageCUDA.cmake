# CUDA GPU Kernels
# CUDA is not compatible with all versions of FindGCC
# Find more information here:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz3x7mwvZrG
# https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
# it was found that you can change the gcc version cuda uses using soft links:
# sudo ln -s /usr/bin/gcc /opt/cuda/bin/gcc 
# sudo ln -s /usr/bin/g++ /opt/cuda/bin/g++
# the compilation.

# in case cuda can not be found it can help to export the path to the toolkit as
# export CUDA_TOOLKIT_ROOT_DIR=/opt/cuda

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# Check if any CUDA compiler is found
if(CMAKE_CUDA_COMPILER)

    # Enable CUDA language
    enable_language(CUDA)

    # Check if NVIDIA compiler
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (NVIDIA)")

        # Set CUDA standard if not already defined
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 14)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        # Set target architectures
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 120)

        # Find the CUDA Toolkit (optional, but useful for libraries like cuBLAS)
        find_package(CUDAToolkit REQUIRED)


        # this is no longer needed as CUDA should be compatible with GCC 15 now
        # maybe at some point this issue will return so we leave the code here

        # # Detect or allow override for CUDA host compiler (g++)
        # set(CUDA_HOST_COMPILER "" CACHE FILEPATH "CUDA host compiler (e.g., /usr/bin/g++-13)")

        # if(NOT CUDA_HOST_COMPILER)
        #     find_program(CUDA_GXX_PATH NAMES g++-13 g++-12 g++-11)
        #     if(CUDA_GXX_PATH)
        #         set(CUDA_HOST_COMPILER ${CUDA_GXX_PATH})
        #         message(STATUS "Detected compatible CUDA host compiler: ${CUDA_HOST_COMPILER}")
        #     else()
        #         message(WARNING "No compatible g++ detected for CUDA. Falling back to system default: ${CMAKE_CXX_COMPILER}")
        #     endif()
        # else()
        #     message(STATUS "Using user-specified CUDA host compiler: ${CUDA_HOST_COMPILER}")
        # endif()

        # # Append -ccbin flag if CUDA_HOST_COMPILER is set
        # if(CUDA_HOST_COMPILER)
        #     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${CUDA_HOST_COMPILER}")
        # endif()

    else()
        message(WARNING "CUDA compiler found, but not NVIDIA: ${CMAKE_CUDA_COMPILER_ID}")
    endif()

else()
    message(STATUS "CUDA not found.")
endif()
