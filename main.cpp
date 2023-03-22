// test.c
#include "app/Gaussian.h"
#include "app/Phi4LFT.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <Eigen/Dense>

const int n = 3;
const double kappa = 2.0; // k = 2 is below transition temp
const double lambda = 1.5;

const Eigen::Matrix<double, 6, 1> mean {{-0.3, 0.4, 0, 0,0,0}};
const Eigen::Matrix<double, 6, 1> var {{1.0, 0.5, 1, 1, 1, 1}};
const double priorWidth = 6;

const double epsilon = 0.001;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 20000;
const double precisionCriterion = 1e-3;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};


void runPhi4()
{
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);

    CHMC sampler = CHMC(likelihood, epsilon, pathLength);
    Logger logger = Logger("Phi4");

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.Initialise();
    NS.Run();
}


void runGaussian() {
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var, priorWidth);

    // RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, epsilon, pathLength);
    Logger logger = Logger("Gaussian");

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.Initialise();
    NS.Run();

}


int main() {
    runPhi4();
}



