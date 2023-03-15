#include "NestedSampler.h"
#include "MockSampler.h"
#include "MockLikelihood.h"
#include "MockLogger.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>


using ::testing::_;
using ::testing::Return;

class NestedSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_CALL(mLikelihood, GetDimension())
                .WillOnce(Return(theta.size()));

        mNestedSampler = std::make_unique<NestedSampler>(mSampler, mLikelihood, mLogger, config);
    }

    Eigen::Vector2d theta {{0, 0}};
    MCPoint point = {
            theta,
            2,
            3
    };

    MockSampler mSampler;
    MockLikelihood mLikelihood;
    MockLogger mLogger;

    const int numLive = 10;
    const int maxIters = 50;
    const double precisionCriterion = 1e-3;
    NSConfig config = {
            numLive,
            maxIters,
            precisionCriterion
    };

    std::unique_ptr<NestedSampler> mNestedSampler;
};


TEST_F(NestedSamplerTest, InitialiseFromPrior) {
    EXPECT_CALL(mLikelihood, PriorTransform(_))
            .Times(numLive);
    EXPECT_CALL(mLikelihood, LogLikelihood(_))
            .Times(numLive);
    EXPECT_CALL(mLogger, WritePoint(_))
            .Times(numLive);

    mNestedSampler->Initialise();
}


TEST_F(NestedSamplerTest, RunNSLoop) {
    EXPECT_CALL(mLikelihood, PriorTransform(_))
            .Times(numLive);
    EXPECT_CALL(mLikelihood, LogLikelihood(_))
            .Times(numLive);
    EXPECT_CALL(mLogger, WritePoint(_))
            .Times(numLive);

    mNestedSampler->Initialise();

    EXPECT_CALL(mLogger, WritePoint(_))
            .Times(maxIters);
    EXPECT_CALL(mSampler, SamplePoint(_, _))
            .WillRepeatedly(Return(point));

    mNestedSampler->Run();
}