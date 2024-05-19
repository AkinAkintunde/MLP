add_library(_MLRunNets SHARED 
            RunNet1.cu)

target_include_directories(_MLRunNets PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/RunNets")