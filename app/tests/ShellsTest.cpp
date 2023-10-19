#include "LikelihoodPlots.h"
#include "UniformPrior.h"
#include "NestedSampler.h"
#include "CHMC.h"
#include "Logger.h"
#include "Shells.h"
#include <Eigen/Dense>


const double priorWidth = 12;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 10000;
const int maxIters = 200000;
const double precisionCriterion = 1e-6;
const double reflectionRateThreshold = 0.9;
const bool logDiagnostics = false;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        reflectionRateThreshold,
        logDiagnostics
};



void runShells(std::string fname) {
    const int d = 10;

    Logger logger = Logger(fname);

    UniformPrior prior = UniformPrior(d, priorWidth);
    Shells likelihood = Shells(d, 2, 0.1, 3.5);

    generateLikelihoodPlot(likelihood, {-10, 10}, {-10, 10});

    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    CHMC sampler = CHMC(prior, likelihood, params, config.reflectionRateThreshold);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}

int main() {
    runShells("gaussian_shells");
}