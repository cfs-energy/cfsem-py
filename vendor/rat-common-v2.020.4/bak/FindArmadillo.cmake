find_package(PkgConfig)
pkg_check_modules(PC_Armadillo QUIET Armadillo)

find_path(Armadillo_INCLUDE_DIR
    NAMES armadillo
    PATHS ${PC_Armadillo_INCLUDE_DIRS}
    HINTS /usr/include
)

# find_library(Armadillo_LIBRARY
#     NAMES armadillo
#     PATHS ${PC_ARMADILLO_LIBRARY_DIR}
#     HINTS /usr/lib
# )

set(Armadillo_VERSION ${PC_Armadillo_VERSION})

mark_as_advanced(Armadillo_FOUND Armadillo_INCLUDE_DIR Armadillo_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
    REQUIRED_VARS Armadillo_INCLUDE_DIR
    VERSION_VAR Armadillo_VERSION
)

if(Armadillo_FOUND)
    #Set include dirs to parent, to enable includes like #include <rapidjson/document.h>
    get_filename_component(Armadillo_INCLUDE_DIRS ${Armadillo_INCLUDE_DIR} DIRECTORY)
endif()

if(Armadillo_FOUND AND NOT TARGET Arma::Armadillo)
    add_library(Arma::Armadillo INTERFACE IMPORTED)
    set_target_properties(Arma::Armadillo PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Armadillo_INCLUDE_DIRS}"
        # INTERFACE_LINK_LIBRARIES "${Armadillo_LIBRARY}"
    )
endif()
