set(LIB_SOURCES
        CHMC.cpp)

add_library(CHMC ${LIB_SOURCES})

target_include_directories(CHMC PUBLIC ${PROJECT_SOURCE_DIR}/include)