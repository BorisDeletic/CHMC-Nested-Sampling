#include "Hamiltonian.h"
#include "MockLikelihood.h"
#include "MockParams.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>

using ::testing::_;
using ::testing::Return;

class HamiltonianTest : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_CALL(likelihood, GetDimension())
                .WillOnce(Return(x.size()));
        EXPECT_CALL(likelihood, LogLikelihood(_))
                .WillOnce(Return(0));

        EXPECT_CALL(likelihood, Gradient(_))
            .WillOnce(Return(zero));

        hamiltonian = std::make_unique<Hamiltonian>(likelihood, params);

        hamiltonian->SetHamiltonian(x, p, likelihoodConstraint);
    }

    Eigen::Vector2d x {{1.0, 1.0}};
    Eigen::Vector2d p {{-1.0, 2.0}};
    Eigen::Vector2d zero {{0, 0}};
    Eigen::Vector2d ones {{1, 1}};

    const double likelihoodConstraint = -2;
    const double epsilon = 0.1;

    MockLikelihood likelihood;
    std::unique_ptr<Hamiltonian> hamiltonian;

    StaticParams params = StaticParams(epsilon, 50, 2);
};


class GaussianHamiltonianTest : public ::testing::Test {
protected:
    void SetUp() override {

    }

    Eigen::Vector2d mean {{0, 0}};
    Eigen::Vector2d var {{0.5, 0.5}};

    Eigen::Vector2d zero {{0, 0}};

    const double epsilon = 0.05;
    const double steps = 500;
    StaticParams params = StaticParams(epsilon, steps, 2);

    GaussianLikelihood mGaussianLikelihood = GaussianLikelihood(mean, var);
};



TEST_F(HamiltonianTest, ZeroGradientEvolve) {
    int steps = 15;

    EXPECT_CALL(likelihood, LogLikelihood(_))
            .Times(steps)
            .WillRepeatedly(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps)
            .WillRepeatedly(Return(zero));

    for (int i = 0; i < steps; i++) {
        hamiltonian->Evolve();
    }

    const Eigen::VectorXd& xf = hamiltonian->GetX();

    EXPECT_DOUBLE_EQ(xf[0], -0.5);
    EXPECT_DOUBLE_EQ(xf[1], 4.0);
}


TEST_F(HamiltonianTest, CircularMotionEvolve) {
    const double tolerance = 1e-3;
    const double k = 0.1;
    int steps = 50;

    EXPECT_CALL(likelihood, LogLikelihood(_))
            .Times(steps + 1)
            .WillRepeatedly(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps + 1)
            .WillRepeatedly([k] (const Eigen::VectorXd& x) {
                return - k * x / pow(x.norm(), 1.5);
            });

    //set init conditions
    Eigen::Vector2d x {{0.0, 1.0}};
    Eigen::Vector2d p {{sqrt(k), 0.0}};
    hamiltonian->SetHamiltonian(x, p, likelihoodConstraint);

    for (int i = 0; i < steps; i++) {
        hamiltonian->Evolve();
    }

    const Eigen::VectorXd& xf = hamiltonian->GetX();

    EXPECT_NEAR(xf.norm(), 1.0, tolerance);
    EXPECT_NEAR(xf[0], 1.0, 0.1);
    EXPECT_NEAR(xf[1], 0.0, 0.1);
}


TEST_F(HamiltonianTest, ReflectPerpendicularIncident) {
    int steps = 2;

    Eigen::Vector2d xi {{1.0, 1.0}};
    Eigen::Vector2d pi {{1.0, 0.0}};
    Eigen::Vector2d grad {{-0.5, 0.0}};

    EXPECT_CALL(likelihood, LogLikelihood(_))
            .Times(steps + 2)
            .WillOnce(Return(-10))
            .WillOnce(Return(-10))
            .WillOnce(Return(0))
            .WillOnce(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps + 1)
            .WillRepeatedly(Return(grad));

    hamiltonian->SetHamiltonian(xi, pi, likelihoodConstraint);
    for (int i = 0; i < steps; i++) {
        hamiltonian->Evolve();
    }

    const Eigen::VectorXd& pf = hamiltonian->GetP();

    EXPECT_DOUBLE_EQ(pf[0], -1.1);
    EXPECT_DOUBLE_EQ(pf[1], 0.0);
}


TEST_F(HamiltonianTest, Reflect45Degrees) {
    int steps = 2;

    Eigen::Vector2d xi {{1.0, 1.0}};
    Eigen::Vector2d pi {{0.5, 0.5}};
    Eigen::Vector2d grad {{-0.5, 0.0}};

    EXPECT_CALL(likelihood, LogLikelihood(_))
            .Times(steps + 2)
            .WillOnce(Return(-10))
            .WillOnce(Return(-10))
            .WillOnce(Return(0))
            .WillOnce(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps + 1)
            .WillRepeatedly(Return(grad));

    hamiltonian->SetHamiltonian(xi, pi, likelihoodConstraint);

    for (int i = 0; i < steps; i++) {
        hamiltonian->Evolve();
    }
    const Eigen::VectorXd& pf = hamiltonian->GetP();

    EXPECT_DOUBLE_EQ(pf[0], -0.6);
    EXPECT_DOUBLE_EQ(pf[1], 0.5);
}


TEST_F(GaussianHamiltonianTest, HardReflectionOffBoundary) {
    Eigen::Vector2d boundary {{0.6, 0.1}};
    double likelihoodConstraint = mGaussianLikelihood.LogLikelihood(boundary);

    Eigen::Vector2d xi {{-0.45, 0.2}};
    Eigen::Vector2d pi {{1.0, 1.0}};

    mHamiltonian.SetHamiltonian(xi, pi, likelihoodConstraint);
    for (int i = 0; i < steps; i++) {
        mHamiltonian.Evolve();
    }
}