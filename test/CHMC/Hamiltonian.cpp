#include "Hamiltonian.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>

using ::testing::_;
using ::testing::Return;
class MockLikelihood : public ILikelihood {
public:
    MOCK_METHOD(double, Likelihood, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(Eigen::VectorXd, Gradient, (const Eigen::VectorXd&), (override));
};


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
    int steps = 10;

    EXPECT_CALL(likelihood, Likelihood(_))
            .Times(steps)
            .WillRepeatedly(Return(0));
    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps)
            .WillRepeatedly(Return(zero));

    hamiltonian.Evolve(steps);


}
