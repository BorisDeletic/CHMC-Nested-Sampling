// test.c
#include "app/Gaussian.h"
#include "Logger.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include <Eigen/Dense>

const Eigen::Vector3d mean {{0.1, 0.2, 0.3}};
const Eigen::Vector3d var {{0.5, 0.5, 0.5}};

const double epsilon = 0.1;
const int pathLength = 10;
const int numLive = 10;

int main() {
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);
    GaussianPrior prior = GaussianPrior(3);

    CHMC sampler = CHMC(likelihood, epsilon, pathLength);
    Logger logger = Logger("Gaussian_dead.txt");

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, numLive);


    NS.Initialise();
    NS.Run(5);
}
