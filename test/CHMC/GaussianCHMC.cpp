#include "CHMC.h"
#include "MockLikelihood.h"
#include <gtest/gtest.h>


using ::testing::_;
using ::testing::Return;

double Gaussian(double x, double m, double v) {
    return exp(- pow(x - m, 2) / (2*v*v)) / (v * sqrt(2 * M_PI));
}

class GaussianCHMCTest : public ::testing::Test {
protected:
    void SetUp() override {
 //       EXPECT_CALL(likelihood, Gradient(_))
  //              .WillOnce(Return(zero));

    }

    Eigen::Vector2d zero {{0, 0}};
    Eigen::Array2d mean {{0.1, 0.1}};
    Eigen::Array2d var {{0.5, 0.5}};

    GaussianLogLikelihood gaussianLikelihood = GaussianLogLikelihood(mean, var);

    MCPoint initPoint = {
            zero,
            gaussianLikelihood.Likelihood(zero)
    };

    const double infLikelihood = 1e9;
    const double epsilon = 0.05;
    const int pathLength = 30;

    CHMC mCHMC = CHMC(gaussianLikelihood, epsilon, pathLength);
};


TEST_F(GaussianCHMCTest, GaussianDistributionNoConstraint) {
    int numSamples = 1000;
    std::vector<MCPoint> samples;
    samples.push_back(initPoint);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = mCHMC.SamplePoint(samples.back(), infLikelihood);
        samples.push_back(newPoint);
    }

    std::map<double, int> histX;
    std::map<double, int> histY;

    for (auto& sample : samples) {
        histX[std::round(sample.theta[0] * 10) / 10]++;
        histY[std::round(sample.theta[1] * 10) / 10]++;
    }

    int cumError = 0;
    int normalisation = numSamples / 10;

    for (auto [x, num] : histX) {
        int expectedFreq = normalisation * Gaussian(x, mean[0], var[0]);
        cumError += abs(expectedFreq - num);
     //   std::cerr << std::setw(2) << x << ' ' << std::string(num/20, '*') << '\n';

    }

    int tolerance = 10 * sqrt(numSamples);
    EXPECT_NEAR(cumError, 0, tolerance);
}

