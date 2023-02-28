#include "NestedSampler.h"
#include "MockSampler.h"
#include "MockPrior.h"
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
                .WillOnce(Return(2));
        EXPECT_CALL(mPrior, GetDimension())
                .WillOnce(Return(2));

        mNestedSampler = std::make_unique<NestedSampler>(mSampler, mPrior, mLikelihood, mLogger, numLive);
    }

    Eigen::Vector2d theta {{0, 0}};
    MCPoint point = {
            theta,
            2,
            3
    };

    MockSampler mSampler;
    MockPrior mPrior;
    MockLikelihood mLikelihood;
    MockLogger mLogger;

    const int numLive = 10;

    std::unique_ptr<NestedSampler> mNestedSampler;
};


TEST_F(NestedSamplerTest, DifferentDimensionsThrows) {
    EXPECT_CALL(mLikelihood, GetDimension())
            .WillOnce(Return(2));
    EXPECT_CALL(mPrior, GetDimension())
            .WillOnce(Return(3));

    EXPECT_THROW({
        NestedSampler wrongSampler = NestedSampler(mSampler, mPrior, mLikelihood, mLogger, numLive);
    }, std::runtime_error);
};


TEST_F(NestedSamplerTest, InitialiseFromPrior) {
    EXPECT_CALL(mPrior, PriorTransform(_))
            .Times(numLive);
    EXPECT_CALL(mLikelihood, Likelihood(_))
            .Times(numLive);

    mNestedSampler->Initialise();
}


TEST_F(NestedSamplerTest, RunNSLoop) {
    const int steps = 5;
    EXPECT_CALL(mPrior, PriorTransform(_))
            .Times(numLive);
    EXPECT_CALL(mLikelihood, Likelihood(_))
            .Times(numLive);

    mNestedSampler->Initialise();

    EXPECT_CALL(mLogger, WriteDeadPoint(_))
            .Times(steps);
    EXPECT_CALL(mSampler, SamplePoint(_, _))
            .Times(steps)
            .WillRepeatedly(Return(point));

    mNestedSampler->Run(steps);
}