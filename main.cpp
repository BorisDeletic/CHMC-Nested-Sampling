// test.c
#include "app/Gaussian.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <Eigen/Dense>

const Eigen::Matrix<double, 6, 1> mean {{-0.3, 0.4, 0, 0,0,0}};
const Eigen::Matrix<double, 6, 1> var {{1.0, 0.5, 1, 1, 1, 1}};
const double priorWidth = 15;

const double epsilon = 0.5;
const int pathLength = 100;

const int numLive = 1000;
const int maxIters = 10000;
const double precisionCriterion = 1e-3;

int main() {
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var, priorWidth);

   // RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, epsilon, pathLength);
    Logger logger = Logger("Gaussian");

    NSConfig config = {
            numLive,
            maxIters,
            precisionCriterion,
    };

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.Initialise();
    NS.Run();
}
