#include "UniformPrior.h"
#include "GaussianPrior.h"
#include "NestedSampler.h"
#include "Logger.h"
#include "Gaussian.h"
#include <Eigen/Dense>

const double priorWidth = 60;

const double epsilon = 0.1;
const int pathLength = 500;

const int numLive = 3000;
const int maxIters = 2000000;
const double precisionCriterion = 1e-8;
const double reflectionRateThreshold = 0.9;
const bool logDiagnostics = true;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        reflectionRateThreshold,
        logDiagnostics
};



void runUniformGaussian(std::string fname) {
    const int d = 10;
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);

    Logger logger = Logger(fname, logDiagnostics);

    UniformPrior prior = UniformPrior(d, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    CHMC sampler = CHMC(prior, likelihood, params, config.reflectionRateThreshold);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

    double analytic = -d * log(priorWidth);
    std::cout << "Analytic evidence value: " << analytic;

}

// this doesnt work because i havent implemented non uniform priors in the energy function
void runNormalGaussian(std::string fname) {
    const int d = 30;
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);

    Logger logger = Logger(fname);

    GaussianPrior prior = GaussianPrior(d, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(epsilon, pathLength, likelihood.GetDimension());

    CHMC sampler = CHMC(prior, likelihood, params, config.reflectionRateThreshold);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}

int main() {
    runUniformGaussian("gaussian_200d_100nlive_r1");
//    runUniformGaussian("gaussian_2d_100nlive_r2");
//    runUniformGaussian("gaussian_2d_100nlive_r3");
//    runUniformGaussian("gaussian_2d_100nlive_r4");
//    runUniformGaussian("gaussian_2d_100nlive_r5");

//    runNormalGaussian("gaussian_normal200d_100nlive_r1");
//    runNormalGaussian("gaussian_normal200d_100nlive_r2");
//    runNormalGaussian("gaussian_normal200d_100nlive_r3");
//    runNormalGaussian("gaussian_normal200d_100nlive_r4");
//    runNormalGaussian("gaussian_normal200d_100nlive_r5");
}

