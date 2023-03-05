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

    }

    Eigen::Vector2d xi {{1.0, 1.0}};
    Eigen::Vector2d pi {{-1.0, 2.0}};
    Eigen::Vector2d a {{-1.0, -1.0}};
    Eigen::Vector2d zero {{0, 0}};

    LeapfrogIntegrator l_integrator = LeapfrogIntegrator(0.1);
};

TEST_F(LeapfrogTest, UpdateIncorrectOrder) {
    l_integrator.UpdateX(xi, pi, a);
    l_integrator.UpdateP(xi, pi, a);

    EXPECT_THROW({
                     l_integrator.UpdateP(xi, pi, a);
                     l_integrator.UpdateX(xi, pi, a);
    }, std::runtime_error);
}

TEST_F(LeapfrogTest, OneStepNoAcceleration) {
    Eigen::Vector2d xf = l_integrator.UpdateX(xi, pi, zero);
    Eigen::Vector2d pf = l_integrator.UpdateP(xi, pi, zero);

    //
    EXPECT_DOUBLE_EQ(xf[0], 0.9);
    EXPECT_DOUBLE_EQ(xf[1], 1.2);

    EXPECT_DOUBLE_EQ(pf[0], -1.0);
    EXPECT_DOUBLE_EQ(pf[1], 2.0);
}


TEST_F(LeapfrogTest, OneStepConstAcceleration) {
    Eigen::Vector2d xf = l_integrator.UpdateX(xi, pi, a);
    Eigen::Vector2d pf = l_integrator.UpdateP(xi, pi, a);

    EXPECT_DOUBLE_EQ(xf[0], 0.895);
    EXPECT_DOUBLE_EQ(xf[1], 1.195);

    EXPECT_DOUBLE_EQ(pf[0], -1.1);
    EXPECT_DOUBLE_EQ(pf[1], 1.9);
}

TEST_F(LeapfrogTest, TimeReversabilityConstAcceleration) {
    const double threshold = 1e-5;
    int steps = 100;

    Eigen::VectorXd x = xi;
    Eigen::VectorXd p = pi;

    for (int i = 0; i < steps; i++) {
        x = l_integrator.UpdateX(x, p, a);
        p = l_integrator.UpdateP(x, p, a);
    }

    // reverse momentum
    l_integrator.ChangeP(p, -p);
    p = -p;

    for (int i = 0; i < steps; i++) {
        x = l_integrator.UpdateX(x, p, a);
        p = l_integrator.UpdateP(x, p, a);
    }

    EXPECT_NEAR(x[0], xi[0], threshold);
    EXPECT_NEAR(x[1], xi[1], threshold);

    EXPECT_NEAR(p[0], -pi[0], threshold);
    EXPECT_NEAR(p[1], -pi[1], threshold);
}

TEST_F(LeapfrogTest, SolveOneSHMStep) {
    const double threshold = 1e-2;
    double k = 0.5;
    Eigen::Vector2d acc;
    Eigen::VectorXd x = xi;
    Eigen::VectorXd p = pi;

    acc = SHM(k, x);
    x = l_integrator.UpdateX(x, p, acc);
    acc = SHM(k, x);
    p = l_integrator.UpdateP(x, p, acc);


    // numbers checked using mathematica for initial conditions.
    EXPECT_NEAR(x[0], 0.89758435399555, threshold);
    EXPECT_NEAR(x[1], 1.1973344164881, threshold);

    EXPECT_NEAR(p[0], -1.0474593852418, threshold);
    EXPECT_NEAR(p[1], 1.9450437392373, threshold);
}


TEST_F(LeapfrogTest, SolveSHM) {
    const double threshold = 0.1;
    int steps = 200;
    double k = 0.5;
    Eigen::Vector2d acc;
    Eigen::VectorXd x = xi;
    Eigen::VectorXd p = pi;

    acc = SHM(k, x);

    for (int i = 0; i < steps; i++) {
        x = l_integrator.UpdateX(x, p, acc);
        acc = SHM(k, x);
        p = l_integrator.UpdateP(x, p, acc);
    }

    // numbers checked using mathematica for initial conditions.
    EXPECT_NEAR(x[0], -1.4191647676261, threshold);
    EXPECT_NEAR(x[1], 2.8234235488545, threshold);

    EXPECT_NEAR(p[0], -0.70212939061419, threshold);
    EXPECT_NEAR(p[1], -0.71703537701198, threshold);
}
