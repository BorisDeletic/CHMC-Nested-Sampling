#include "Phi4LFT.h"
#include "Phi4Likelihood.h"
#include "Logger.h"
#include "Adapter.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include <iostream>

const int n = 10;
const double kappa = 0.0; // k = 2 is below transition temp
const double lambda = 1.5;

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


int main() {
    runPhi4();
    std::cout << "help";
}