pybind11_add_module(pymlfmm pymlfmm.cpp)
target_link_libraries(pymlfmm PRIVATE rat-mlfmm-st)
target_link_libraries(pymlfmm PRIVATE ${LIBS})

if(ENABLE_CUDA)
set_target_properties(pymlfmm PROPERTIES LINKER_LANGUAGE CUDA)
endif()