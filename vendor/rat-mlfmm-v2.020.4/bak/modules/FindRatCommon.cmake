# include libfind
include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(RAT_COMMON_PKGCONF RAT_COMMON)

# Include dir
find_path(RAT_COMMON_INCLUDE_DIR
  NAMES common/elements.hh common/extra.hh common/parfor.hh
  PATHS ${RAT_COMMON_PKGCONF_INCLUDE_DIR}
  HINTS /usr/local/include/rat
)

# Finally the library itself
find_library(RAT_COMMON_LIBRARY
  NAMES rat-common-st
  PATHS ${RAT_COMMON_PKGCONF_LIBRARY_DIR}
  HINTS /usr/local/lib/rat/common
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(RAT_COMMON_PROCESS_INCLUDES RAT_COMMON_INCLUDE_DIR)
set(RAT_COMMON_PROCESS_LIBS RAT_COMMON_LIBRARY)
libfind_process(RAT_COMMON)

