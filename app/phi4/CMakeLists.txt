# app/CMakeLists.txt

find_package(Eigen3)

add_executable(Phi4LFT
        LFT.cpp LFT.h
        Phi4Likelihood.cpp Phi4Likelihood.h
		PosteriorAnalysis.cpp PosteriorAnalysis.h
        )

add_executable(Phi4Benchmark
		Benchmark.cpp Benchmark.h
		Phi4Likelihood.cpp Phi4Likelihood.h
		)


target_include_directories(Phi4LFT PRIVATE
        ${PROJECT_SOURCE_DIR}/include)

target_include_directories(Phi4Benchmark PRIVATE
		${PROJECT_SOURCE_DIR}/include)


target_link_libraries(Phi4LFT
		CHMC_NS_Lib
		Eigen3::Eigen
		)

target_link_libraries(Phi4Benchmark
		CHMC_NS_Lib
		Eigen3::Eigen
		)
