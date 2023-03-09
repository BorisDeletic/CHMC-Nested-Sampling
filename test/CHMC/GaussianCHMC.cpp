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

    for (const auto [x, num] : hist) {
        const double expectedFreq = 0.1 * samples * Gaussian(x, mean, var);
        cumError += abs(num - expectedFreq);
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
    Eigen::Array2d var {{0.5, 0.5}};
    Eigen::Array<double, 1, 1> mean1 {{0}};
    Eigen::Array<double, 1, 1> var1 {{1.0}};

    GaussianLikelihood gaussianLikelihood = GaussianLikelihood(mean, var);
    GaussianLikelihood gaussian1DLikelihood = GaussianLikelihood(mean1, var1);

    MCPoint initPoint = {
            zero,
            gaussianLikelihood.LogLikelihood(zero),
            1e30
    };

    const double inf = 1e9;
    const double epsilon = 0.001;
    const int pathLength = 50;
    std::pair<double, double> noBound {-inf, inf};


    CHMC mCHMC = CHMC(gaussianLikelihood, epsilon, pathLength);
};


TEST_F(GaussianCHMCTest, GaussianDistributionNoConstraint) {
    Eigen::Vector2d boundary {{0.6, 0.1}};
    double likelihoodConstraint = gaussianLikelihood.LogLikelihood(boundary);

    int numSamples = 5000;

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


    int tolerance = 10 * sqrt(numSamples);
    //EXPECT_NEAR(cumError, 0, tolerance);
}



TEST_F(GaussianCHMCTest, SamplesDontViolateConstraint) {
    Eigen::Vector2d boundary {{0.6, 0.1}};
    double likelihoodConstraint = gaussianLikelihood.LogLikelihood(boundary);

    std::cerr << likelihoodConstraint << std::endl;

    const int numSamples = 500;
    std::vector<MCPoint> samples;
    samples.push_back(initPoint);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = mCHMC.SamplePoint(samples.back(), likelihoodConstraint);
        samples.push_back(newPoint);
        std::cerr << newPoint.likelihood << ", "
            << newPoint.theta[0] << ", "
            << newPoint.theta[1] << std::endl ;
    }

    for (auto& point : samples) {
        EXPECT_GE(point.likelihood, likelihoodConstraint);
    }
}


TEST_F(GaussianCHMCTest, Gaussian1DWithConstraint) {
    int numSamples = 1000;
    Eigen::Matrix<double, 1, 1> boundary {{0.7}};
    double likelihoodConstraint = gaussian1DLikelihood.LogLikelihood(boundary);

    CHMC chmc = CHMC(gaussian1DLikelihood, 0.1, 100);

    MCPoint first = {
            mean1,
            gaussian1DLikelihood.LogLikelihood(mean1),
            likelihoodConstraint
    };

    std::map<double, int> histX;

    std::vector<MCPoint> samples;
    samples.push_back(first);

    for (int i = 0; i < numSamples; i++) {
        MCPoint newPoint = chmc.SamplePoint(samples.back(), likelihoodConstraint);
        samples.push_back(newPoint);
        histX[std::round(newPoint.theta[0] * 10) / 10]++;
    }

    std::cerr << "error = " << CalculateError(histX, mean1[0], var1[0], noBound);

    LogHist(histX);
    int tolerance = 10 * sqrt(numSamples);
    // EXPECT_NEAR(cumError, 0, tolerance);
}
