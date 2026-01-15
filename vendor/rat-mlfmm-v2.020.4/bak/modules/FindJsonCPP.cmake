# include libfind
include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(JSONCPP_PKGCONF JSONCPP)

# Include dir
find_path(JSONCPP_INCLUDE_DIR
  NAMES json/json.h
  PATHS ${JSONCPP_PKGCONF_INCLUDE_DIR}
  HINTS /usr/local/include
)

# Finally the library itself
find_library(JSONCPP_LIBRARY
  NAMES jsoncpp
  PATHS ${JSONCPP_PKGCONF_LIBRARY_DIR}
  HINTS /usr/local/lib/rat/common
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(JSON_CPP_PROCESS_INCLUDES JSONCPP_INCLUDE_DIR)
set(JSON_CPP_PROCESS_LIBS JSONCPP_LIBRARY)
libfind_process(JSONCPP)
