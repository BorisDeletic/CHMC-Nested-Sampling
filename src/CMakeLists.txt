# src/CMakeLists.txt

set(LIB_SOURCES
        enzyme_test.cpp
        CHMC/CHMC.cpp)

add_library(CHMC_NS_Lib ${LIB_SOURCES})

target_include_directories(CHMC_NS_Lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
