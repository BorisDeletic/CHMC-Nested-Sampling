#include "Benchmark.h"
#include "Phi4Likelihood.h"
#include "Logger.h"
#include "Adapter.h"
#include "CHMC.h"
#include "NestedSampler.h"
#include "types.h"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <chrono>

std::string dir = "benchmark";

const double priorWidth = 6;

const double epsilon = 0.1;
const int pathLength = 100;

const int numLive = 100;
const int maxIters = 1000;
const double precisionCriterion = 0.001;

NSConfig config = {
        numLive,
        maxIters,
        precisionCriterion,
};


void runPhi4(std::string fname, int n, double kappa, double lambda)
{
    Phi4Likelihood likelihood = Phi4Likelihood(n, kappa, lambda, priorWidth);
    Logger logger = Logger(fname);

    Adapter params = Adapter(epsilon, pathLength, n*n);

    CHMC sampler = CHMC(likelihood, params);

    NestedSampler NS = NestedSampler(sampler, likelihood, logger, config);

    NS.SetAdaption(&params);
    NS.Initialise();
    NS.Run();
}


void logResults(std::string fname, int n, long long time) {
    std::ofstream file;
    file.open(fname, std::ios_base::app);

    file << n << "," << n*n << "," << maxIters << "," << time << "," << (double)time/maxIters << std::endl;

    file.close();
}


void generateDimensionalityBenchmark() {

    const int nMin = 810;
    const int nMax = 1010;
    const double lambda = 0.03;
    const double kappa = 0.11750;

    if (!std::filesystem::is_directory(dir) || !std::filesystem::exists(dir)) { // Check if src folder exists
        std::filesystem::create_directory(dir); // create src folder
    }

    for (int n = nMin; n < nMax; n += 10) {
        std::ostringstream fname;
        fname << dir;
        fname << "/benchmark";

        std::cout << std::endl << std::endl <<"Running Phi4 Benchmark: n=" << n << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        runPhi4(fname.str(), n, kappa, lambda);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        std::cout << "time=" << dt << "ms" << std::endl;

        fname << ".txt";
        logResults(fname.str(), n, dt);
    }
}



int main() {
    generateDimensionalityBenchmark();
}
