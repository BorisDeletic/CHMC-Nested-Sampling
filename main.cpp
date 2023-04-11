#include "app/Gaussian.h"
#include "app/Phi4LFT.h"
#include "app/TopologicalTrap.h"
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
const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n*n);
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



void runPhi4()
{
    Logger logger = Logger("Phi4");
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);

    //StaticParams params = StaticParams(n*n);
    Adapter params = Adapter(epsilon, pathLength, n*n);

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



void generateContours(ILikelihood& likelihood, std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("isocontours.dat");

    for (double i = xran.first; i < xran.second; i += 0.05) {
        for (double j = yran.first; j < yran.second; j += 0.05) {
            Eigen::VectorXd theta(likelihood.GetDimension());
            theta[0] = i;
            theta[1] = j;
            const double like = likelihood.LogLikelihood(theta);

            file << std::setprecision(3) << std::fixed << i << " " << j << " " <<
                 std::setprecision(4) << std::fixed << like << std::endl;
 //           file << (int)(i*10) << " " << (int)(j*10) << " " << like << std::endl;
        }
        file << std::endl;
    }

    file.close();
}


void generateGradientField(ILikelihood& likelihood, std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("gradient.dat");

    for (double i = xran.first; i < xran.second; i += 0.4) {
        for (double j = yran.first; j < yran.second; j += 0.4) {
            Eigen::VectorXd theta(likelihood.GetDimension());
            theta[0] = i;
            theta[1] = j;

            Eigen::VectorXd grad = likelihood.Gradient(theta).normalized() / 6;
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
    Phi4Likelihood phiLikelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);
//    TopologicalTrap topoLikelihood = TopologicalTrap(2);

    std::pair<float, float> xran = {-4, 4};
    std::pair<float, float> yran = {-4, 4};
    generateContours(phiLikelihood, xran, yran);
    generateGradientField(phiLikelihood, xran, yran);
}

int main() {
    //generateLikelihoodPlot();
    runPhi4();
 //   runGaussian();
 //   runTopoTrap();
}



