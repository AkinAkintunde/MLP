cmake_minimum_required(VERSION 3.8...3.18)
project (MLP LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

add_subdirectory(pybind11)
add_subdirectory(Nets)
add_subdirectory(Propagators)
add_subdirectory(Checker)

pybind11_add_module(MLP Main.cpp)

set_target_properties(MLP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(MLP PRIVATE 
                      _MLNets
                      _MLPropagators
                      _MLChecker)

enable_language(CUDA)