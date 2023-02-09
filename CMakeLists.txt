cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED)

message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# C and C++ Compilers can only be configured before project(...)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${PROJECT_SOURCE_DIR}/cmake")

project(CHMC-Nested-Sampling)

# Define LLVM and Clang compiler toolchains
#set(LLVM_MIN_SUPPORTED 7.1.0)
#set(LLVM_MAX_SUPPORTED 12.1.0)
#
## Find LLVM
#if (DEFINED LLVM_DIR)
#    set(LLVM_HINTS HINTS ${LLVM_DIR} "${LLVM_DIR}/lib/cmake/llvm" "${LLVM_DIR}/cmake")
#endif()
#
#find_package(LLVM REQUIRED CONFIG ${LLVM_HINTS})
#
#include_directories(${LLVM_INCLUDE_DIRS})
#add_definitions(${LLVM_DEFINITIONS})
#
#if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER LLVM_MAX_SUPPORTED)
#    message(FATAL_ERROR "Found unsupported version: LLVM ${LLVM_PACKAGE_VERSION};\nPlease set LLVM_DIR pointing to the llvm version ${LLVM_MIN_SUPPORTED} to ${LLVM_MAX_SUPPORTED} build or installation folder")
#endif()
#
#message(STATUS "Found supported version: LLVM ${LLVM_PACKAGE_VERSION}")
#message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
#
#
##Find Clang
#if (DEFINED CLANG_DIR)
#    set(CLANG_HITS HINTS ${CLANG_DIR} "${CLANG_DIR}/lib/cmake/clang" "${CLANG_DIR}/cmake")
#endif()

#find_package(Clang REQUIRED CONFIG ${CLANG_HINTS})

#set(LLVM_EXTERNAL_LIT "/home/boris/.local/bin/lit")

#find_package(Enzyme REQUIRED)
#message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
#message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# Find Enzyme Package
find_package(Enzyme REQUIRED COMPONENTS ClangEnzyme)

get_target_property(ClangEnzyme ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR} LOCATION)

message("Found Enzyme at: " ${ClangEnzyme})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -load -Xclang ${ClangEnzyme} -O2 -fno-vectorize -fno-unroll-loops")

option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(extern/googletest)
endif()

add_subdirectory(src)

add_executable(CHMC-Nested-Sampling main.cpp)

