#include "likelihoods/Gaussian.h"
#include "likelihoods/TopologicalTrap.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <Eigen/Dense>

const int n = 10;
const double kappa = 0.0; // k = 2 is below transition temp
const double lambda = 1.5;

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


void runTopoTrap() {
    int d = 4;
    Logger logger = Logger("TopoTrap");
    TopologicalTrap likelihood = TopologicalTrap(d);

    // StaticParams params = StaticParams(likelihood.GetDimension());
    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, params);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();
}




int main() {
    //generateLikelihoodPlot();
 //   runGaussian();
 //   runTopoTrap();
}



