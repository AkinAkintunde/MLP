add_library(_MLChecker SHARED 
            Checker.cpp)

target_include_directories(_MLChecker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Checker")