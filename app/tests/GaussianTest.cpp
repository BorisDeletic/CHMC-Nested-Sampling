#include "GaussianTest.h"
#include "UniformPrior.h"
#include "NestedSampler.h"
#include "Logger.h"
#include "Gaussian.h"
#include <Eigen/Dense>

const int d = 200;
const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);
const double priorWidth = 20;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 200000;
const double precisionCriterion = 1e-2;
const double reflectionRateThreshold = 0.9;
const bool logDiagnostics = false;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        reflectionRateThreshold,
        logDiagnostics
};



void runGaussian() {
    Logger logger = Logger("Gaussian");

    UniformPrior prior = UniformPrior(d, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    CHMC sampler = CHMC(prior, likelihood, params, config.reflectionRateThreshold);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}


int main() {
    runGaussian();
}