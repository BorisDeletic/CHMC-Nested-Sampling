#include "Hamiltonian.h"
#include "../MockLikelihood.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>

using ::testing::_;
using ::testing::Return;

class HamiltonianTest : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_CALL(likelihood, Gradient(_))
            .WillOnce(Return(zero));

        hamiltonian.SetHamiltonian(x, p, likelihoodConstraint);
    }

    Eigen::Vector2d x {{1.0, 1.0}};
    Eigen::Vector2d p {{-1.0, 2.0}};
    Eigen::Vector2d zero {{0, 0}};

    const double likelihoodConstraint = 1e9;

    MockLikelihood likelihood;
    Hamiltonian hamiltonian = Hamiltonian(likelihood, 0.1);
};


TEST_F(HamiltonianTest, ZeroGradientEvolve) {
    int steps = 15;

    EXPECT_CALL(likelihood, Likelihood(_))
            .Times(steps)
            .WillRepeatedly(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps)
            .WillRepeatedly(Return(zero));

    hamiltonian.Evolve(steps);
    const Eigen::VectorXd& xf = hamiltonian.GetX();

    EXPECT_DOUBLE_EQ(xf[0], -0.5);
    EXPECT_DOUBLE_EQ(xf[1], 4.0);
}


TEST_F(HamiltonianTest, CircularMotionEvolve) {
    const double tolerance = 1e-3;
    const double k = 0.1;
    int steps = 50;

    EXPECT_CALL(likelihood, Likelihood(_))
            .Times(steps)
            .WillRepeatedly(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps + 1)
            .WillRepeatedly([k] (const Eigen::VectorXd& x) {
                return - k * x / pow(x.norm(), 1.5);
            });

    //set init conditions
    Eigen::Vector2d x {{0.0, 1.0}};
    Eigen::Vector2d p {{sqrt(k), 0.0}};
    hamiltonian.SetHamiltonian(x, p, likelihoodConstraint);

    hamiltonian.Evolve(steps);

    const Eigen::VectorXd& xf = hamiltonian.GetX();

    EXPECT_NEAR(xf.norm(), 1.0, tolerance);
    EXPECT_NEAR(xf[0], 1.0, 0.1);
    EXPECT_NEAR(xf[1], 0.0, 0.1);
}
