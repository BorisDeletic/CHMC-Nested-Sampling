#include "LFT.h"
#include "UniformPrior.h"
#include "Phi4Likelihood.h"
#include "../LikelihoodPlots.h"
#include "Logger.h"
#include "Adapter.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

std::string phase_dir = "phase_diagram";
std::string correlation_dir = "correlation";

const bool logDiagnostics = false;
const double priorWidth = 4;

const double epsilon = 0.01;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 500000;
const double precisionCriterion = 1e-1;
const double reflectionRateTarget = 0.05;
const double acceptRateTarget = 0.8;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        logDiagnostics
};


void runPhi4(std::string fname, int n, double kappa, double lambda)
{
    UniformPrior prior = UniformPrior(n*n, priorWidth);
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda);
    Logger logger = Logger(fname, logDiagnostics);

    Adapter params = Adapter(n*n, epsilon, pathLength, reflectionRateTarget, acceptRateTarget);

    CHMC sampler = CHMC(prior, likelihood, params);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();
}


void generatePhaseDiagramData() {
    const int n = 32;
    double kappaMax = 0.5;
    double lambdaMax = 0.1;
    int resolution = 50;

    if (!std::filesystem::is_directory(phase_dir) || !std::filesystem::exists(phase_dir)) { // Check if src folder exists
        std::filesystem::create_directory(phase_dir); // create src folder
    }

    for (double k = 0; k < kappaMax; k += kappaMax / resolution) {
        for (double l = 0; l < lambdaMax; l += lambdaMax / resolution) {
            std::ostringstream fname;
            fname << phase_dir;
            fname << "/Phi4_" << std::setprecision(5) << std::fixed << k << "_" << l;
            
            if(std::filesystem::exists(fname.str() + ".stats"))
                continue;

            std::cout << std::endl << std::endl <<"Running Phi4: kappa=" << k << ", lambda=" << l << std::endl;

            runPhi4(fname.str(), n,k, l);
        }
    }
}

void generatePhaseData() {
    const int n = 32;
    double kappaMin = 0.16;
    double kappaMax = 0.24;
    double lambda = 0.1;
    int resolution = 10;

    if (!std::filesystem::is_directory(phase_dir) || !std::filesystem::exists(phase_dir)) { // Check if src folder exists
        std::filesystem::create_directory(phase_dir); // create src folder
    }

    for (double k = kappaMin; k < kappaMax; k += (kappaMax - kappaMin) / resolution) {
        std::ostringstream fname;
        fname << phase_dir;
        fname << "/Phi4_" << std::setprecision(5) << std::fixed << k << "_" << lambda;

        if(std::filesystem::exists(fname.str() + ".stats"))
            continue;

        std::cout << std::endl << std::endl <<"Running Phi4: kappa=" << k << ", lambda=" << lambda << std::endl;

        runPhi4(fname.str(), n,k, lambda);
    }
}


void generateCorrelationData() {

    if (!std::filesystem::is_directory(correlation_dir) || !std::filesystem::exists(correlation_dir)) { // Check if src folder exists
        std::filesystem::create_directory(correlation_dir); // create src folder
    }

    const int n = 64;
    const double lambda = 0.03;
    const double kappaMin = 0.11900;
    const double kappaMax = 0.119500;
    const double resolution = 1;

    for (double k = kappaMin; k < kappaMax; k += (kappaMax - kappaMin) / resolution) {
        std::ostringstream fname;
        fname << correlation_dir;
        fname << "/Phi4_" << std::setprecision(5) << std::fixed << k << "_" << lambda;

        if(std::filesystem::exists(fname.str() + ".stats"))
            continue;

        std::cout << std::endl << std::endl <<"Running Phi4 for Correlations: kappa=" << k << ", lambda=" << lambda << std::endl;

        runPhi4(fname.str(), n,k, lambda);
    }
}




int main() {

//    Phi4Likelihood likelihood = Phi4Likelihood(2, 0.05, 1.5, priorWidth);
//    generateLikelihoodPlot(likelihood, {-4, 4}, {-4, 4});
    generatePhaseDiagramData();
//    generatePhaseData();

 //  generateCorrelationData();
//    runPhi4("Phi4_posterior_sampling", 32, 0.25, 0.02);
    std::cout << "help!";
}
