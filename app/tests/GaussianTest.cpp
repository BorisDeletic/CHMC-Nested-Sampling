#include "GaussianTest.h"


const int d = 50;
const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);
const double priorWidth = 6;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 20000;
const double precisionCriterion = 1e-2;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};



void runGaussian() {
    Logger logger = Logger("Gaussian");
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var, priorWidth);

    // StaticParams params = StaticParams(likelihood.GetDimension());
    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, params);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}


main