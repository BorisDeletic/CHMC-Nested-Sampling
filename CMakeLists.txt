cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED)
message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

## C and C++ Compilers can only be configured before project(...)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${PROJECT_SOURCE_DIR}/cmake")


project(CHMC_Nested_Sampling)

option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
    add_subdirectory(extern/googletest)
endif()

add_subdirectory(extern/eigen)

# Find Enzyme Package
find_package(Enzyme REQUIRED COMPONENTS ClangEnzyme)

get_target_property(ClangEnzyme ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR} LOCATION)
message("Found Enzyme at: " ${ClangEnzyme})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}
    -Xclang -load -Xclang ${ClangEnzyme} -O2 -fno-vectorize -fno-unroll-loops")


add_executable(CHMC_Nested_Sampling main.cpp)

add_subdirectory(src)
add_subdirectory(likelihood)

target_include_directories(CHMC_Nested_Sampling PUBLIC
        include
        likelihood)

target_link_libraries(CHMC_Nested_Sampling
        CHMC_NS_Lib
        Likelihood)

