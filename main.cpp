// test.c
#include "app/Gaussian.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include <Eigen/Dense>

const Eigen::Vector2d mean {{-0.3, 0.4}};
const Eigen::Vector2d var {{0.5, 0.5}};

const double epsilon = 0.1;
const int pathLength = 10;
const int numLive = 1000;

int main() {
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);
    GaussianPrior prior = GaussianPrior(2);

    RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    Logger logger = Logger("Gaussian_dead-birth.txt");

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, numLive);


    NS.Initialise();
    NS.Run(5000);
}
