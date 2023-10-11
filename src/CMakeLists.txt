# src/CMakeLists.txt

find_package(Eigen3)

set(LIB_SOURCES
        CHMC/CHMC.cpp
        CHMC/LeapfrogIntegrator.cpp
        CHMC/Hamiltonian.cpp
        CHMC/Adapter.cpp
        RejectionSampler.cpp
        Logger.cpp
        NestedSampler.cpp
        Clustering.cpp
        )

add_library(CHMC_NS_Lib ${LIB_SOURCES})

target_link_libraries(CHMC_NS_Lib
        Eigen3::Eigen)

target_include_directories(CHMC_NS_Lib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CHMC
        )
