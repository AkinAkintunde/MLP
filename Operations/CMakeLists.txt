add_library(_MLOperations SHARED 
            VecAdd.cu)

target_include_directories(_MLOperations PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Operations")