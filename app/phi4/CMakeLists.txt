# app/CMakeLists.txt

find_package(Eigen3)

add_executable(Phi4LFT
        LFT.cpp LFT.h
        Phi4Likelihood.cpp Phi4Likelihood.h
		PosteriorAnalysis.cpp PosteriorAnalysis.h
        )

target_include_directories(Phi4LFT PRIVATE
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(Phi4LFT
		CHMC_NS_Lib
		Eigen3::Eigen
	)
