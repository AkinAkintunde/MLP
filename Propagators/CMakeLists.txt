add_library(_MLPropagators SHARED 
            Propagator1.cu
            Propagator1.cpp)

target_include_directories(_MLPropagators PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Propagators")