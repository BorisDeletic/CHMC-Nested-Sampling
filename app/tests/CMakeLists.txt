find_package(Eigen3)

set(LIKELIHOOD_SOURCES
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Gaussian.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Phi4Likelihood.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/TopologicalTrap.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/UniformPrior.cpp
        )

add_executable(GaussianTest
        GaussianTest.cpp
        ${LIKELIHOOD_SOURCES}
        )


target_include_directories(GaussianTest PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/app/likelihoods
        )


target_link_libraries(GaussianTest
        CHMC_NS_Lib
        Eigen3::Eigen
        )
