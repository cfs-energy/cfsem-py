# Create Debian release version from current date. DPKG only allows
# insalling higher versions of an already installed package by default,
# unless forced. This allows us to install newly generated packages
# without changing the project version number.
execute_process(COMMAND date +%y%m%d%H%M%S OUTPUT_VARIABLE RELEASE_VER)
string(REGEX REPLACE "\n$" "" RELEASE_VER "${RELEASE_VER}")

set(CPACK_PACKAGE_VENDOR "Project RAT")
set(CPACK_PACKAGE_CONTACT "saman@saman-gh.co.uk")
set(CPACK_CPACK_DEBIAN_PACKAGE_RELEASE "${RELEASE_VER}")
set(CPACK_DEBIAN_PACKAGE_VERSION "${PROJECT_VERSION}-${CPACK_CPACK_DEBIAN_PACKAGE_RELEASE}")

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
set(CPACK_DEBIAN_PACKAGE_NAME "libratmlfmm")

# Make sure premissions are correct
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)

include(CPack)
