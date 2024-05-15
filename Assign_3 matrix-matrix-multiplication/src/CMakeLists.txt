# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add your library
add_library(studentlib main.cpp)

# Set include directories
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Set compile options including OpenMP flags
target_compile_options(studentlib PRIVATE -Ofast -march=native ${OpenMP_CXX_FLAGS})

# Link OpenMP library
target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
