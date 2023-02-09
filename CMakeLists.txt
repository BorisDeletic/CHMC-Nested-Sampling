cmake_minimum_required(VERSION 3.7...3.25)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(CHMC-Nested-Sampling)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
    add_subdirectory(extern/googletest)
endif()

#add_subdirectory(extern/enzyme/enzyme)
add_subdirectory(src)

add_executable(CHMC-Nested-Sampling main.cpp)
