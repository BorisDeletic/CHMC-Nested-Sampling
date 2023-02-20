#include "LeapfrogIntegrator.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <stdexcept>

Eigen::VectorXd SHM(const double k, const Eigen::VectorXd& x) {
    return - k * x;
}

class LeapfrogTest : public ::testing::Test {
protected:
    void SetUp() override {
        l_integrator.SetX(x);
        l_integrator.SetP(p);
    }

    Eigen::Vector2d x {{1.0, 1.0}};
    Eigen::Vector2d p {{-1.0, 2.0}};
    Eigen::Vector2d a {{-1.0, -1.0}};
    Eigen::Vector2d zero {{0, 0}};

    LeapfrogIntegrator l_integrator = LeapfrogIntegrator(0.1);
};

TEST_F(LeapfrogTest, UpdateIncorrectOrder) {
    l_integrator.UpdateX(a);
    l_integrator.UpdateP(a);

    EXPECT_THROW({
                     l_integrator.UpdateP(a);
                     l_integrator.UpdateX(a);
    }, std::runtime_error);
}

TEST_F(LeapfrogTest, OneStepNoAcceleration) {
    l_integrator.UpdateX(zero);
    l_integrator.UpdateP(zero);

    Eigen::Vector2d xf = l_integrator.GetX();
    Eigen::Vector2d pf = l_integrator.GetP();

    //
    EXPECT_DOUBLE_EQ(xf[0], 0.9);
    EXPECT_DOUBLE_EQ(xf[1], 1.2);

    EXPECT_DOUBLE_EQ(pf[0], -1.0);
    EXPECT_DOUBLE_EQ(pf[1], 2.0);
}


TEST_F(LeapfrogTest, OneStepConstAcceleration) {
    l_integrator.UpdateX(a);
    l_integrator.UpdateP(a);

    Eigen::Vector2d xf = l_integrator.GetX();
    Eigen::Vector2d pf = l_integrator.GetP();

    EXPECT_DOUBLE_EQ(xf[0], 0.895);
    EXPECT_DOUBLE_EQ(xf[1], 1.195);

    EXPECT_DOUBLE_EQ(pf[0], -1.1);
    EXPECT_DOUBLE_EQ(pf[1], 1.9);
}

TEST_F(LeapfrogTest, TimeReversabilityConstAcceleration) {
    const double threshold = 1e-5;
    int steps = 100;

    for (int i = 0; i < steps; i++) {
        l_integrator.UpdateX(a);
        l_integrator.UpdateP(a);
    }

    Eigen::Vector2d pReverse = l_integrator.GetP();
    l_integrator.SetP(-pReverse);

    for (int i = 0; i < steps; i++) {
        l_integrator.UpdateX(a);
        l_integrator.UpdateP(a);
    }

    Eigen::Vector2d xf = l_integrator.GetX();
    Eigen::Vector2d pf = l_integrator.GetP();

    EXPECT_NEAR(xf[0], x[0], threshold);
    EXPECT_NEAR(xf[1], x[1], threshold);

    EXPECT_NEAR(pf[0], -p[0], threshold);
    EXPECT_NEAR(pf[1], -p[1], threshold);
}

TEST_F(LeapfrogTest, SolveOneSHMStep) {
    const double threshold = 1e-2;
    double k = 0.5;
    Eigen::Vector2d acc;
    acc = SHM(k, l_integrator.GetX());

    l_integrator.UpdateX(acc);
    acc = SHM(k, l_integrator.GetX());
    l_integrator.UpdateP(acc);

    Eigen::Vector2d xf = l_integrator.GetX();
    Eigen::Vector2d pf = l_integrator.GetP();

    // numbers checked using mathematica for initial conditions.
    EXPECT_NEAR(xf[0], 0.89758435399555, threshold);
    EXPECT_NEAR(xf[1], 1.1973344164881, threshold);

    EXPECT_NEAR(pf[0], -1.0474593852418, threshold);
    EXPECT_NEAR(pf[1], 1.9450437392373, threshold);
}


TEST_F(LeapfrogTest, SolveSHM) {
    const double threshold = 0.1;
    int steps = 200;
    double k = 0.5;
    Eigen::Vector2d acc;

    acc = SHM(k, l_integrator.GetX());

    for (int i = 0; i < steps; i++) {
        l_integrator.UpdateX(acc);
        acc = SHM(k, l_integrator.GetX());
        l_integrator.UpdateP(acc);
    }

    Eigen::Vector2d xf = l_integrator.GetX();
    Eigen::Vector2d pf = l_integrator.GetP();

    // numbers checked using mathematica for initial conditions.
    EXPECT_NEAR(xf[0], -1.4191647676261, threshold);
    EXPECT_NEAR(xf[1], 2.8234235488545, threshold);

    EXPECT_NEAR(pf[0], -0.70212939061419, threshold);
    EXPECT_NEAR(pf[1], -0.71703537701198, threshold);
}
