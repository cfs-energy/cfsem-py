find_package(PkgConfig)
pkg_check_modules(PC_superlu QUIET superlu)

find_path(superlu_INCLUDE_DIRS
  NAMES supermatrix.h
  HINTS ${SUPERLU_DIR}/include ${PC_superlu_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
  PATH_SUFFIXES superlu)

find_library(superlu_LIBRARIES
  NAMES libsuperlu superlu
  HINTS ${superlu_DIR}/lib ${PC_superlu_LIBRARIES} /usr/lib /usr/local/lib /opt/local/lib)

# Extract version from pkg-config if available
if (PC_superlu_VERSION)
    set(superlu_VERSION ${PC_superlu_VERSION})
endif()

# If pkg-config didn't provide a version, extract it from the header
if (NOT superlu_VERSION AND superlu_INCLUDE_DIRS)
    file(STRINGS "${superlu_INCLUDE_DIRS}/slu_Cnames.h" SUPERLU_VERSION_LINE REGEX "#define SUPERLU_[A-Z_]+[0-9]+")
    
    if (SUPERLU_VERSION_LINE)
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" superlu_VERSION ${SUPERLU_VERSION_LINE})
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(superlu
  REQUIRED_VARS superlu_LIBRARIES superlu_INCLUDE_DIRS
  VERSION_VAR superlu_VERSION)

# Enforce EXACT version matching
if(superlu_FIND_VERSION_EXACT)
    if(superlu_VERSION AND NOT superlu_VERSION VERSION_EQUAL superlu_FIND_VERSION)
        message(STATUS "SuperLU version ${superlu_VERSION} found, but EXACT version ${superlu_FIND_VERSION} is required!")
        set(superlu_FOUND FALSE)
    endif()
endif()

# Create an imported target if found and correct version
if (superlu_FOUND AND NOT TARGET superlu)
  add_library(superlu INTERFACE IMPORTED)
  set_target_properties(superlu PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${superlu_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${superlu_LIBRARIES}"
  )
endif()
