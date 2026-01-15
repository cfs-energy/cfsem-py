find_package(PkgConfig)
pkg_check_modules(PC_JsonCPP QUIET JsonCPP)

find_path(JsonCPP_INCLUDE_DIR
	NAMES json/config.h
	PATHS ${PC_JsonCPP_INCLUDE_DIRS}
	HINTS $ENV{JSONCPP}/include /usr/local/include /opt/local/include /usr/include/jsoncpp 
)

find_library(JsonCPP_LIBRARY
	NAMES jsoncpp
	PATHS ${PC_JsonCPP_LIBRARY_DIR}
	HINTS $ENV{JSONCPP}/lib /usr/local/lib /opt/local/lib /usr/lib 
)

set(JsonCPP_VERSION ${PC_JsonCPP_VERSION})

mark_as_advanced(JsonCPP_FOUND JsonCPP_INCLUDE_DIR JsonCPP_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JsonCPP
	REQUIRED_VARS JsonCPP_INCLUDE_DIR JsonCPP_LIBRARY
	VERSION_VAR JsonCPP_VERSION
)

# if(JsonCPP_FOUND)
# 	#Set include dirs to parent, to enable includes like #include <rapidjson/document.h>
# 	get_filename_component(JsonCPP_INCLUDE_DIRS ${JsonCPP_INCLUDE_DIR} DIRECTORY)
# endif()

if(JsonCPP_FOUND AND NOT TARGET JsonCPP::JsonCPP)
	add_library(JsonCPP::JsonCPP INTERFACE IMPORTED)
	set_target_properties(JsonCPP::JsonCPP PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${JsonCPP_INCLUDE_DIR}"
		INTERFACE_LINK_LIBRARIES "${JsonCPP_LIBRARY}"
	)
endif()
