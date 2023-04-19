#include "Phi4LFT.h"
#include "Phi4Likelihood.h"
#include "LikelihoodPlots.h"
#include "Logger.h"
#include "Adapter.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

std::string phase_dir = "phase_diagram";

const int n = 32;
//const double kappa = 0.01; // k = 2 is below transition temp
//const double lambda = 0.2;

const double priorWidth = 6;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 20000;
const double precisionCriterion = 10;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};



void runPhi4(std::string fname, double kappa, double lambda)
{
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);
    Logger logger = Logger(fname);

    Adapter params = Adapter(epsilon, pathLength, n*n);

    CHMC sampler = CHMC(likelihood, params);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();
}


void generatePhaseDiagramData() {
    double kappaMax = 0.5;
    double lambdaMax = 0.03;
    int resolution = 40;

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

            runPhi4(fname.str(), k, l);
        }
    }
}






int main() {
  //  generateLikelihoodPlot(likelihood, {-2, 2}, {-2, 2});
    generatePhaseDiagramData();
  //  runPhi4();
    std::cout << "help";
}
