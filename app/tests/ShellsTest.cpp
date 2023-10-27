#include "LikelihoodPlots.h"
#include "UniformPrior.h"
#include "NestedSampler.h"
#include "CHMC.h"
#include "Logger.h"
#include "Shells.h"
#include <Eigen/Dense>


const double priorWidth = 12;

const double epsilon = 0.1;
const int pathLength = 5000;

const int numLive = 1000;
const int maxIters = 500000;
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
    const int d = 20;

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

    std::cout << "This run D = " << d << std::endl;
    std::cout << "Analytic evidences:" << std::endl;
    std::cout << "D = 2: logZ = Z_true = -1.75" << std::endl;
    std::cout << "D = 5: logZ = Z_true = -5.67" << std::endl;
    std::cout << "D = 10: logZ = Z_true = -14.59" << std::endl;
    std::cout << "D = 20: logZ = Z_true = -36.09" << std::endl;
    std::cout << "D = 30: logZ = Z_true = -60.13" << std::endl;
}

int main() {
    runShells("gaussian_shells");
}