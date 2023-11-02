find_package(Eigen3)
find_package(OpenMP)

set(LIKELIHOOD_SOURCES
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Gaussian.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Phi4Likelihood.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/TopologicalTrap.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/UniformPrior.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/GaussianPrior.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Eggbox.cpp
        ${PROJECT_SOURCE_DIR}/app/likelihoods/Shells.cpp
        ${PROJECT_SOURCE_DIR}/app/LikelihoodPlots.cpp
        )

add_executable(GaussianTest
        GaussianTest.cpp
        ${LIKELIHOOD_SOURCES}
        )

add_executable(EggboxTest
        EggboxTest.cpp
        ${LIKELIHOOD_SOURCES}
        )

add_executable(ShellsTest
        ShellsTest.cpp
        ${LIKELIHOOD_SOURCES}
        )




target_include_directories(GaussianTest PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/app/likelihoods
        )

target_include_directories(EggboxTest PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/app/likelihoods
        ${PROJECT_SOURCE_DIR}/app
        )

target_include_directories(ShellsTest PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/app/likelihoods
        ${PROJECT_SOURCE_DIR}/app
        )


target_link_libraries(GaussianTest PRIVATE
        CHMC_NS_Lib
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        )

target_link_libraries(EggboxTest
        CHMC_NS_Lib
        Eigen3::Eigen
        )

target_link_libraries(ShellsTest
        CHMC_NS_Lib
        Eigen3::Eigen
        )
