# src/CMakeLists.txt

set(LIB_SOURCES
        enzyme_test.cpp)

add_library(CHMC_NS_Lib ${LIB_SOURCES})

add_subdirectory(CHMC)

target_include_directories(CHMC_NS_Lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(CHMC_NS_Lib PRIVATE CHMC)