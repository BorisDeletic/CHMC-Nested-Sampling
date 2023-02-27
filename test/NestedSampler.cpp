#include "CHMC.h"
#include "MockPrior.h"
#include "MockLikelihood.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "NestedSampler.h"


class MockCHMC : public CHMC {
public:
    MockCHMC(ILikelihood& l) : CHMC(l, 0.0, 0) {}
    MOCK_METHOD(const MCPoint, SamplePoint, (const MCPoint&, double), (override));
};


using ::testing::_;
using ::testing::Return;

class NestedSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        //       EXPECT_CALL(likelihood, Gradient(_))
        //              .WillOnce(Return(zero));
    }

    MockPrior mPrior;
    MockLikelihood mLikelihood;
    MockCHMC mCHMC = MockCHMC(mLikelihood);

    const int numLive = 10;
    std::string name  = "NS_Test";
    NestedSampler mNestedSampler = NestedSampler(mCHMC, mPrior, mLikelihood, numLive, name);
};


TEST_F(NestedSamplerTest, FirstTest) {
    mNestedSampler.Run(10);
};