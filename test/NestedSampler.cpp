#include "NestedSampler.h"
#include "MockSampler.h"
#include "MockPrior.h"
#include "MockLikelihood.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>


using ::testing::_;
using ::testing::Return;

class NestedSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_CALL(mLikelihood, GetDimension())
                .WillOnce(Return(2));
        EXPECT_CALL(mPrior, GetDimension())
                .WillOnce(Return(2));

        mNestedSampler = std::make_unique<NestedSampler>(mSampler, mPrior, mLikelihood, numLive, name);
    }

    MockSampler mSampler;
    MockPrior mPrior;
    MockLikelihood mLikelihood;

    const int numLive = 10;
    std::string name  = "NS_Test";

    std::unique_ptr<NestedSampler> mNestedSampler;
};


TEST_F(NestedSamplerTest, DifferentDimensionsThrows) {
    EXPECT_CALL(mLikelihood, GetDimension())
            .WillOnce(Return(2));
    EXPECT_CALL(mPrior, GetDimension())
            .WillOnce(Return(3));

    EXPECT_THROW({
        NestedSampler wrongSampler = NestedSampler(mSampler, mPrior, mLikelihood, numLive, name);
    }, std::runtime_error);
};

