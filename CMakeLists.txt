cmake_minimum_required(VERSION 3.9)

set(USE_ENZYME TRUE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


IF(USE_ENZYME)
    find_package(Enzyme REQUIRED)

    message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
    message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

    ## C and C++ Compilers can only be configured before project(...)
    set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
    set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF()


project(CHMC_Nested_Sampling)


# Find Enzyme Package
IF(USE_ENZYME)
    find_package(Enzyme REQUIRED COMPONENTS ClangEnzyme)

    get_target_property(ClangEnzyme ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR} LOCATION)
    message("Found Enzyme at: " ${ClangEnzyme})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}
        -Xclang -load -Xclang ${ClangEnzyme} -O2 -fno-vectorize -fno-unroll-loops")

    add_subdirectory(enzyme)
ENDIF()

#$<$<CONFIG:RELEASE>:-Ofast>
add_compile_options(
        $<$<CONFIG:RELEASE>:-funroll-loops>
        $<$<CONFIG:DEBUG>:-O0>
        $<$<CONFIG:DEBUG>:-ggdb3>
)


add_subdirectory(extern/eigen)
add_subdirectory(extern/googletest)
add_subdirectory(test)
add_subdirectory(src)
add_subdirectory(app/phi4)

add_executable(CHMC_Nested_Sampling
        main.cpp
        app/likelihoods/Gaussian.cpp
        app/phi4/Phi4Likelihood.cpp
        app/likelihoods/TopologicalTrap.cpp
        app/likelihoods/Rosenbrock.cpp
        app/LikelihoodPlots.cpp
        )

target_include_directories(CHMC_Nested_Sampling PUBLIC
        include
        include/CHMC
        app
        )

target_link_libraries(CHMC_Nested_Sampling
        CHMC_NS_Lib
        )

if(USE_ENZYME)
    target_include_directories(CHMC_Nested_Sampling PUBLIC
            enzyme)

    target_link_libraries(CHMC_Nested_Sampling
            EnzymeLikelihoods)
ENDIF()