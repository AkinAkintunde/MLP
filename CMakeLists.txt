cmake_minimum_required(VERSION 3.4...3.18)
project (MLP LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

add_subdirectory(pybind11)
add_subdirectory(Activations)
add_subdirectory(Nets)
add_subdirectory(Propagators)

add_library(_MLP SHARED 
            Main.cu)

pybind11_add_module(MLP Main.cu)

set_target_properties(MLP PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 

target_link_libraries(MLP PUBLIC 
                      _MLP 
                      _MLActivations
                      _MLNets
                      _MLPropagators)

enable_language(CUDA)