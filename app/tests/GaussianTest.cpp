#include "UniformPrior.h"
#include "GaussianPrior.h"
#include "NestedSampler.h"
#include "Logger.h"
#include "Gaussian.h"
#include <Eigen/Dense>
#include <omp.h>

const double priorWidth = 60;

const double epsilon = 0.1;
const int pathLength = 500;
const double reflectionRateTarget = 0.01;

const int numLive = 3000;
const int maxIters = 2000000;
const double precisionCriterion = 1e-8;
const bool logDiagnostics = true;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
        logDiagnostics
};



void runGaussian(std::string fname) {
    const int d = 10;
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);

    Logger logger = Logger(fname, logDiagnostics);

    UniformPrior prior = UniformPrior(d, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(likelihood.GetDimension(), epsilon, pathLength, reflectionRateTarget);

    CHMC sampler = CHMC(prior, likelihood, params);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

    double analytic = -d * log(priorWidth);
    std::cout << "Analytic evidence value: " << analytic;

}


void runUniformGaussian(std::string out_name, int dimension, int numPoints, int repetition) {
    int maxIterations = dimension * numPoints * 1000;

    NSConfig gaussianConfig = {
            numPoints,
            maxIterations,
            precisionCriterion,
            logDiagnostics
    };

    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(dimension);
    const Eigen::VectorXd var  = Eigen::VectorXd::Ones(dimension);

    std::string fname = "gaussian_batch/" + std::to_string(dimension) + "D_"
            + std::to_string(numPoints) + "nlive_r" + std::to_string(repetition);

    Logger logger = Logger(fname, logDiagnostics);

    UniformPrior prior = UniformPrior(dimension, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(likelihood.GetDimension(), epsilon, pathLength, reflectionRateTarget);

    CHMC sampler = CHMC(prior, likelihood, params);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, gaussianConfig);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

    double analytic = -dimension * log(priorWidth);
    std::cout << "Analytic evidence value: " << analytic << std::endl;

    std::ofstream mOutFile;
    mOutFile.open(out_name, std::ios_base::app);

    NSInfo summary = NS.GetInfo();

    //dimension,num_live,path_length,reflect_rate,iters,logZ,std_logZ,true_logZ
    mOutFile << dimension << ",";
    mOutFile << summary.numLive << ",";
    mOutFile << pathLength << ",";
    mOutFile << reflectionRateTarget << ",";
    mOutFile << summary.iter << ",";
    mOutFile << summary.meanLogZ << ",";
    mOutFile << summary.stdLogZ << ",";
    mOutFile << analytic << std::endl;

    mOutFile.close();
}


// this doesnt work because i havent implemented non uniform priors in the energy function
void runNormalGaussian(std::string fname) {
    const int d = 30;
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);

    Logger logger = Logger(fname);

    GaussianPrior prior = GaussianPrior(d, priorWidth);
    GaussianLikelihood likelihood = GaussianLikelihood(mean, var);

    Adapter params = Adapter(likelihood.GetDimension(), epsilon, pathLength, reflectionRateTarget);

    CHMC sampler = CHMC(prior, likelihood, params);

    NestedSampler NS = NestedSampler(sampler, prior, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();

}


void runGaussianBatch() {
    int maxDim = 200;
    int repetitions = 5;

    std::string fname = "gaussian_batch.csv";
    std::ofstream mOutFile;
    mOutFile.open(fname);

    mOutFile << "dimension,num_live,path_length,reflect_rate,iters,logZ,std_logZ,true_logZ" << std::endl;
    mOutFile.close();

    for (double d = 5; d < maxDim; d *= 1.5) {
        int numPoints = 20 * floor(d);

        #pragma omp parallel
        {
            for (int i = 0; i < repetitions; i++) {
                runUniformGaussian(fname, floor(d), numPoints, i);
            }
        }
    }
}


int main() {
    runGaussianBatch();

//    runGaussian("gaussian_200d_100nlive_r1");
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

