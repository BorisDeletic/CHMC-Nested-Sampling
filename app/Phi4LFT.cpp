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


const int n = 10;
const double kappa = 0.01; // k = 2 is below transition temp
const double lambda = 0.2;

const double priorWidth = 6;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 20000;
const double precisionCriterion = 0.05;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};


void generatePhaseDiagramData() {
    double kappaMax = 0.5;
    double lambdaMax = 0.02;
    int resolution = 10;

    for (double k = 0; k < kappaMax; k += kappaMax / resolution) {
        for (double l = 0; l < lambdaMax; l += lambdaMax / resolution) {
            std::ostringstream fname;
            fname << "phase_diagram/Phi4_" << std::setprecision(3) << std::fixed << k << "_" << l;
            if(std::filesystem::exists(fname.str() + ".stats"))
                continue;

            std::cout << std::endl << std::endl <<"Running Phi4: kappa=" << k << ", lambda=" << l << std::endl;

            Phi4Likelihood likelihood = Phi4Likelihood(n, k, l, priorWidth);
            Logger logger = Logger(fname.str());

            Adapter params = Adapter(epsilon, pathLength, n*n);

            CHMC sampler = CHMC(likelihood, params);

            NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

            NS.SetAdaption(&params);
            NS.Initialise();
            NS.Run();
        }
    }

}


void runPhi4()
{
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);
    Logger logger = Logger("Phi4");

    Adapter params = Adapter(epsilon, pathLength, n*n);

    CHMC sampler = CHMC(likelihood, params);
    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();
}





int main() {
  //  generateLikelihoodPlot(likelihood, {-2, 2}, {-2, 2});
    generatePhaseDiagramData();
  //  runPhi4();
    std::cout << "help";
}
