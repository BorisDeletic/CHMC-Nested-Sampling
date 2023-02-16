include(GoogleTest)

add_executable(CMHC_Test
        first_test.cpp
#        CHMC/Leapfrog.cpp
)

target_link_libraries(CMHC_Test
    #    GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(CMHC_Test)
