#include "CHMC.h"
#include "MockLikelihood.h"
#include <gtest/gtest.h>


using ::testing::_;
using ::testing::Return;

double Gaussian(double x, double m, double v) {
    return exp(- pow(x - m, 2) / (2*v*v)) / (v * sqrt(2 * M_PI));
}

void LogHist(std::map<double, int>& hist) {
    double norm = 0;
    for (const auto [x, num] : hist) { norm += num/300.0f ; }
    for (const auto [x, num] : hist) {
        std::cerr << std::setw(2) << x << ' ' << std::string(num/norm, '*') << '\n';
    }
}

double CalculateError(std::map<double, int>& hist, double mean, double var, std::pair<double, double> bounds) {
    int cumError = 0;
    int samples  = 0;
    for (const auto [x, num] : hist) { samples += num; }

    double boundsNormalization = 1.0 - 0.5 * erfc((mean - bounds.first) * M_SQRT1_2 / var);
    boundsNormalization -= 0.5 * erfc((bounds.second - mean) * M_SQRT1_2 / var);

    for (const auto [x, num] : hist) {
        const double expectedFreq = 0.1 * samples * Gaussian(x, mean, var) / boundsNormalization;
        cumError += abs(num - expectedFreq);
//        std::cerr << abs(num - expectedFreq) << std::endl;
    }

    return (float)cumError / samples;
}

class GaussianCHMCTest : public ::testing::Test {
protected:
    void SetUp() override {
 //       EXPECT_CALL(likelihood, Gradient(_))
  //              .WillOnce(Return(zero));

    }

    Eigen::Vector2d zero {{0, 0}};
    Eigen::Array2d mean {{0.1, 0.1}};
    Eigen::Array2d var {{0.2, 1.0}};

    GaussianLikelihood gaussianLikelihood = GaussianLikelihood(mean, var);

    MCPoint initPoint = {
            zero,
            gaussianLikelihood.LogLikelihood(zero),
            1e30
    };

    const double epsilon = 0.01;
    const int pathLength = 100;
    std::pair<double, double> noBound {-DBL_MAX, DBL_MAX};

    CHMC mCHMC = CHMC(gaussianLikelihood, epsilon, pathLength);
};


TEST_F(GaussianCHMCTest, GaussianDistributionNoConstraint) {
    mCHMC.WarmupAdapt(initPoint);
    mCHMC.WarmupAdapt(initPoint);

    int numSamples = 5000;

    std::map<double, int> histX;
    std::map<double, int> histY;

    std::vector<MCPoint> samples;
    samples.push_back(initPoint);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = mCHMC.SamplePoint(samples.back(), -DBL_MAX);
        samples.push_back(newPoint);

        histX[std::round(newPoint.theta[0] * 10) / 10]++;
        histY[std::round(newPoint.theta[1] * 10) / 10]++;
    }

  //  LogHist(histX);
    const double errX = CalculateError(histX, mean[0], var[0], noBound);
    const double errY = CalculateError(histY, mean[1], var[1], noBound);

    EXPECT_LE(errX, 0.1);
    EXPECT_LE(errY, 0.1);
}


TEST_F(GaussianCHMCTest, SamplesDontViolateConstraint) {
    Eigen::Vector2d boundary {{0.5, 0.5}};
    double likelihoodConstraint = gaussianLikelihood.LogLikelihood(boundary);

    const int numSamples = 500;
    std::vector<MCPoint> samples;
    samples.push_back(initPoint);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = mCHMC.SamplePoint(samples.back(), likelihoodConstraint);
        samples.push_back(newPoint);
    }

    for (auto& point : samples) {
        EXPECT_GE(point.likelihood, likelihoodConstraint);
    }
}


TEST_F(GaussianCHMCTest, CorrectDistributionWithConstraint) {
    mCHMC.WarmupAdapt(initPoint);
    mCHMC.WarmupAdapt(initPoint);

    Eigen::Vector2d bound{{0.5, 0.1}};
    const double likelihoodConstraint = gaussianLikelihood.LogLikelihood(bound);

    int numSamples = 10000;

    std::map<double, int> histX;
    std::map<double, int> histY;

    std::vector<MCPoint> samples;
    samples.push_back(initPoint);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = mCHMC.SamplePoint(samples.back(), likelihoodConstraint);
        samples.push_back(newPoint);

        histX[std::round(newPoint.theta[0] * 10) / 10]++;
        histY[std::round(newPoint.theta[1] * 10) / 10]++;
    }

    //LogHist(histX);
    const double errX = CalculateError(histX, mean[0], var[0], {-0.3, 0.5});
    const double errY = CalculateError(histY, mean[1], var[1], {-1.9, 2.0});

    EXPECT_LE(errX, 0.1);
    EXPECT_LE(errY, 0.1);

}