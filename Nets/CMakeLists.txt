add_library(_MLNets SHARED 
            Net1.cpp
            Net1.cu
            Strides/Stride1.cu
            Strides/Stride1.cpp)

target_include_directories(_MLNets PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Nets")
