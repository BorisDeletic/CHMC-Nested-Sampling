#include "app/Gaussian.h"
#include "app/Phi4LFT.h"
#include "Logger.h"
#include "RejectionSampler.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <Eigen/Dense>

const int n = 10;
const double kappa = 2.0; // k = 2 is below transition temp
const double lambda = 1.5;

const int d = 50;
const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n*n);
const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);
const double priorWidth = 6;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 500;
const int maxIters = 20000;
const double precisionCriterion = 1e-3;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};


class StaticParams : public IParams {
public:
    StaticParams(int dims) : mMetric(Eigen::VectorXd::Ones(dims)) {}

    double GetEpsilon() override { return epsilon; };
    int GetPathLength() override { return pathLength; };
    const Eigen::VectorXd& GetMetric() override { return mMetric; };

private:
    const Eigen::VectorXd mMetric;
};


void runPhi4()
{
    Logger logger = Logger("Phi4");
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);

    //StaticParams params = StaticParams(n*n);
    Adapter adaptiveParams = Adapter(epsilon, pathLength, ones);

    CHMC sampler = CHMC(likelihood, adaptiveParams);
    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&adaptiveParams);
    NS.Initialise();
    NS.Run();
}


void runGaussian() {
    Logger logger = Logger("Gaussian");
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var, priorWidth);

   // StaticParams params = StaticParams(likelihood.GetDimension());
    Adapter params = Adapter(epsilon, pathLength, Eigen::VectorXd::Ones(likelihood.GetDimension()));

    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);
    CHMC sampler = CHMC(likelihood, params);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}


int main() {
    runPhi4();
//    runGaussian();
}



