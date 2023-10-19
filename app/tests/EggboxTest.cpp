#include "EggboxTest.h"
#include "LikelihoodPlots.h"
#include "UniformPrior.h"
#include "NestedSampler.h"
#include "CHMC.h"
#include "Logger.h"
#include "Eggbox.h"
#include <Eigen/Dense>


const double priorWidth = 2;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 1000;
const int maxIters = 50000;
const double precisionCriterion = 1e-5;
const double reflectionRateThreshold = 0.9;
const bool logDiagnostics = false;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        reflectionRateThreshold,
        logDiagnostics
};



void runEggbox(std::string fname) {
    const int d = 2;

    Logger logger = Logger(fname);

    UniformPrior prior = UniformPrior(d, priorWidth);
    Eggbox likelihood = Eggbox(d);

    generateLikelihoodPlot(likelihood, {0, 1}, {0, 1});

    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    CHMC sampler = CHMC(prior, likelihood, params, config.reflectionRateThreshold);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}

int main() {
    runEggbox("eggbox");
}