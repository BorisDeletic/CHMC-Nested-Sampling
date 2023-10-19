# app/CMakeLists.txt

find_package(Eigen3)

set(LIKELIHOOD_SOURCES
        likelihoods/Gaussian.cpp
        likelihoods/Phi4Likelihood.cpp
        likelihoods/TopologicalTrap.cpp
        likelihoods/UniformPrior.cpp
        likelihoods/GaussianPrior.cpp
        likelihoods/Eggbox.cpp
        )


add_executable(Phi4LFT
        phi4/LFT.cpp
        phi4/PosteriorAnalysis.cpp
        LikelihoodPlots.cpp
        ${LIKELIHOOD_SOURCES}
        )

add_executable(Phi4Benchmark
        phi4/Benchmark.cpp phi4/Benchmark.h
        ${LIKELIHOOD_SOURCES}
        )




target_include_directories(Phi4LFT PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include)

target_include_directories(Phi4Benchmark PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include)


target_link_libraries(Phi4LFT
        CHMC_NS_Lib
        Eigen3::Eigen
        )

target_link_libraries(Phi4Benchmark
        CHMC_NS_Lib
        Eigen3::Eigen
        )

add_subdirectory(tests)