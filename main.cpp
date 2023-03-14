// test.c
#include "app/Gaussian.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <Eigen/Dense>

const Eigen::Vector2d mean {{-0.3, 0.4}};
const Eigen::Vector2d var {{1.0, 0.5}};
const double priorWidth = 15;

const double epsilon = 0.01;
const int pathLength = 500;

const int numLive = 1000;
const int maxIters = 20000;
const double precisionCriterion = 1e-3;

int main() {
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);
    GaussianPrior prior = GaussianPrior(mean.size(), priorWidth);

   // RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, epsilon, pathLength);
    Logger logger = Logger("Gaussian");

    NSConfig config = {
            numLive,
            maxIters,
            precisionCriterion,
    };

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);


    NS.Initialise();
    NS.Run();
}
