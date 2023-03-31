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
const double lambda = 3.5;

const int d = 50;
const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n*n);
const Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
const Eigen::VectorXd var  = Eigen::VectorXd::Ones(d);
const double priorWidth = 6;

const double epsilon = 0.01;
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
    Adapter params = Adapter(epsilon, pathLength, ones);

    CHMC sampler = CHMC(likelihood, params);
    //RejectionSampler sampler = RejectionSampler(likelihood, epsilon);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
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


void generateContours(std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("isocontours.dat");

    Phi4Likelihood likelihood = Phi4Likelihood(2, kappa, lambda, priorWidth);
    for (double i = xran.first; i < xran.second; i += 0.01) {
        for (double j = yran.first; j < yran.second; j += 0.01) {
            Eigen::Vector4d theta {{i, j, 0, 0}};
            const double like = likelihood.LogLikelihood(theta);

            file << std::setprecision(3) << std::fixed << i << " " << j << " " <<
                 std::setprecision(4) << std::fixed << like << std::endl;
 //           file << (int)(i*10) << " " << (int)(j*10) << " " << like << std::endl;
        }
        file << std::endl;
    }

    file.close();
}


void generateGradientField(std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("gradient.dat");

    Phi4Likelihood likelihood = Phi4Likelihood(2, kappa, lambda, priorWidth);
    for (double i = xran.first; i < xran.second; i += 0.2) {
        for (double j = yran.first; j < yran.second; j += 0.2) {
            Eigen::Vector4d theta {{i, j, 0, 0}};
            Eigen::Vector4d grad = likelihood.Gradient(theta).normalized() / 10;
         //   Eigen::Vector4d grad = likelihood.Gradient(theta) / 500;

            file << std::setprecision(3) << std::fixed << i << " " << j << " " <<
                 std::setprecision(4) << std::fixed << grad[0] << " " << grad[1] << std::endl;
            //           file << (int)(i*10) << " " << (int)(j*10) << " " << like << std::endl;

        }
        file << std::endl;
    }

    file.close();
}


void generateLikelihoodPlot() {
    std::pair<float, float> xran = {-2.3, 2.3};
    std::pair<float, float> yran = {-2.3, 2.3};
    generateContours(xran, yran);
    generateGradientField(xran, yran);
}

int main() {
  //  generateLikelihoodPlot();
    runPhi4();
 //   runGaussian();
}



