enable_testing()
include(GoogleTest)

add_executable(CHMC_Test
        CHMC/LeapfrogIntegrator.cpp
)

target_include_directories(CHMC_Test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/CHMC)

target_link_libraries(CHMC_Test
        GTest::gtest_main
        CHMC_NS_Lib
)

gtest_discover_tests(CHMC_Test)
