find_package(PkgConfig)
pkg_check_modules(PC_RatCMN QUIET RatCMN)

find_path(RatCMN_INCLUDE_DIR
    NAMES rat/common/extra.hh
    PATHS ${PC_RatCMN_INCLUDE_DIRS}
)

set(RatCMN_VERSION ${PC_RatCMN_VERSION})

mark_as_advanced(RatCMN_FOUND RatCMN_INCLUDE_DIR RatCMN_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RatCMN
    REQUIRED_VARS RatCMN_INCLUDE_DIR
    VERSION_VAR RatCMN_VERSION
)

if(RatCMN_FOUND)
    #Set include dirs to parent, to enable includes like #include <rapidjson/document.h>
    get_filename_component(RatCMN_INCLUDE_DIRS ${RatCMN_INCLUDE_DIR} DIRECTORY)
endif()

if(RatCMN_FOUND AND NOT TARGET Rat::Common)
    add_library(Rat::Common INTERFACE IMPORTED)
    set_target_properties(Rat::Common PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${RatCMN_INCLUDE_DIRS}"
    )
endif()
