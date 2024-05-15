cmake_minimum_required(VERSION 3.18)
project(studentlib LANGUAGES CUDA)

# Locate CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Add library target
add_library(studentlib main.cu)

# Set include directories
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Enable separable compilation
set_target_properties(studentlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compiler and linker options
target_compile_options(studentlib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>             # Maximum optimization level
    $<$<COMPILE_LANGUAGE:CUDA>:-march=native>   # Generate code optimized for the native architecture
)

target_link_options(studentlib PRIVATE
    $<$<CUDA_LINKER_LANGUAGE:CUDA>:-Xlinker -rpath>      # Add runtime library search paths
)
