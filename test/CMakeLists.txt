include(GoogleTest)


add_executable(first_test
        first_test.cpp
)

target_link_libraries(first_test
        GTest::gtest_main
)

gtest_discover_tests(first_test)
