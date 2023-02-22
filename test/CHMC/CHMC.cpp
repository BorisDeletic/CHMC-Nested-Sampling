#include "CHMC.h"
#include "../MockLikelihood.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>


using ::testing::_;
using ::testing::Return;


class CHMCTest : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_CALL(likelihood, Gradient(_))
                .WillOnce(Return(zero));
    }

    Eigen::Vector2d zero {{0, 0}};

    MCPoint initPoint = {

    };

    MockLikelihood likelihood;

    double epsilon = 0.1;
    int pathLength = 50;
    CHMC mCHMC = CHMC(likelihood, epsilon, pathLength);
};


TEST_F(CHMCTest, GaussianLikelihood) {
    int steps = 15;

    EXPECT_CALL(likelihood, Likelihood(_))
            .Times(steps)
            .WillRepeatedly(Return(0));

    EXPECT_CALL(likelihood, Gradient(_))
            .Times(steps)
            .WillRepeatedly(Return(zero));


}

