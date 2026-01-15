# include libfind
include(LibFindMacros)

# Dependencies

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(NVBLAS_PKGCONF NVBLAS)

# Include dir
find_path(NVBLAS_INCLUDE_DIR
  NAMES nvblas.h
  HINTS /opt/cuda/include/
  PATHS ${NVBLAS_PKGCONF_INCLUDE_DIR}
)

# Finally the library itself
find_library(NVBLAS_LIBRARY
  NAMES nvblas
  HINTS /opt/cuda/lib64/
  PATHS ${NVBLAS_PKGCONF_LIBRARY_DIR}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(NVBLAS_PROCESS_INCLUDES NVBLAS_INCLUDE_DIR)
set(NVBLAS_PROCESS_LIBS NVBLAS_LIBRARY)
libfind_process(NVBLAS)
