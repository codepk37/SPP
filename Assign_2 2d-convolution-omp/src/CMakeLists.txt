# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add your library
add_library(studentlib main.cpp)

# Include directories
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Compile options including OpenMP support
target_compile_options(studentlib PRIVATE -Ofast -march=native -mfma -O3 -funroll-loops -g ${OpenMP_CXX_FLAGS})

# Link OpenMP library
target_link_libraries(studentlib PRIVATE OpenMP::OpenMP_CXX)
