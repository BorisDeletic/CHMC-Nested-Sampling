enable_testing()
include(GoogleTest)


add_executable(CHMC_Test
        LeapfrogIntegrator.cpp
        Hamiltonian.cpp
        GaussianCHMC.cpp
        )


target_include_directories(CHMC_Test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/CHMC
        ${PROJECT_SOURCE_DIR}/test/mock
        )

target_link_libraries(CHMC_Test
        GTest::gtest_main
        GTest::gmock_main
        CHMC_NS_Lib
        )

gtest_discover_tests(CHMC_Test)
