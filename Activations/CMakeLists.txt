add_library(_MLActivations SHARED 
            Activation1.cpp)

target_include_directories(_MLActivations PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Activations")