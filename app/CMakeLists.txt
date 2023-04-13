
add_executable(Phi4LFT
        Phi4LFT.cpp Phi4LFT.h
        LikelihoodPlots.cpp LikelihoodPlots.h
        likelihoods/Phi4Likelihood.cpp likelihoods/Phi4Likelihood.h
        )

target_include_directories(Phi4LFT PRIVATE
        likelihoods
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(Phi4LFT
        CHMC_NS_Lib)