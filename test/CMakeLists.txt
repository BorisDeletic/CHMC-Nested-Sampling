enable_testing()
include(GoogleTest)


add_executable(NestedSampler_Test
        NestedSampler.cpp
        )

target_include_directories(NestedSampler_Test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CHMC
        ${PROJECT_SOURCE_DIR}/test/mock
        )

target_link_libraries(NestedSampler_Test
        GTest::gtest_main
        GTest::gmock_main
        CHMC_NS_Lib
)


add_subdirectory(CHMC)

gtest_discover_tests(NestedSampler_Test)
