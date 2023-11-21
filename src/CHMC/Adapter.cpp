#include "Adapter.h"

Adapter::Adapter(const int dimension, double initEpsilon, int initPathLength, double reflectRateTarget, double acceptRateTarget)
    :
    mDimension(dimension),
    mEpsilon(initEpsilon),
    mPathLength(initPathLength),
    mReflectionRateTarget(reflectRateTarget),
    mAcceptRateTarget(acceptRateTarget),
    mMetric(Eigen::VectorXd::Ones(mDimension))
{
    mMu = log(initEpsilon);
    mIter = 0;
    mSBar = 0;
    mXBar = 0;
}

void Adapter::Restart() {
    mIter = 0;
    mSBar = 0;
    mXBar = 0;
}

void Adapter::AdaptEpsilon(double reflectRate, double acceptProb) {
    mIter++;

    acceptProb = acceptProb > 1 ? 1 : acceptProb;

    // Nesterov Dual-Averaging of log(epsilon)
    const double eta = 1.0 / (mIter + mT0);

//    mSBar = (1.0 - eta) * mSBar + eta * (reflectRate - mDelta);
    if (reflectRate < mReflectionRateTarget) {
        mSBar = (1.0 - eta) * mSBar + eta * (mAcceptRateTarget - acceptProb);
    } else {
        mSBar = (1.0 - eta) * mSBar + eta * (reflectRate - mReflectionRateTarget);
    }

    const double x = mMu - mSBar * std::sqrt(mIter) / mGamma;
    const double x_eta = std::pow(mIter, -mKappa);

    mXBar = (1.0 - x_eta) * mXBar + x_eta * x;

    mEpsilon = std::exp(x);
}


void Adapter::AdaptMetric(const std::multiset<MCPoint> &livePoints) {

    const int size = livePoints.size();
    Eigen::ArrayXd likelihood(size);

    int i = 0;
    for (const auto& point : livePoints) {
        likelihood[i] = point.likelihood;
        i++;
    }

    Eigen::ArrayXd centered = likelihood - likelihood.mean();

    const double var = centered.pow(2).sum() / size;

    // Scale metric alpha.
    mAlpha = sqrt(2 * var) / mDimension;
    //  made metric smaller

    mMetric = mAlpha * Eigen::VectorXd::Ones(mDimension);

    //std::cout << "alpha = " << mAlpha << std::endl;
}


/*
bool CHMC::WarmupAdapt(const MCPoint &init)
{
    int dim = init.theta.size();
    Eigen::MatrixXd samples( mWarmupSteps, dim);

    samples.row(0) = init.theta;

    for (int i = 1; i < mWarmupSteps; i++)
    {
        MCPoint last = {
                samples.row(i-1),
                0,
                0};

        MCPoint next = SamplePoint(last, -DBL_MAX);
        samples.row(i) = next.theta;
    }

    Eigen::VectorXd newMetric = CalculateVar(samples).cwiseInverse();
    // std::cerr << std::endl << newMetric << std::endl;

    mHamiltonian.SetMetric(newMetric);

    return true;
}


Eigen::VectorXd CHMC::CalculateVar(const Eigen::MatrixXd& samples) {
    Eigen::MatrixXd centered = samples.rowwise() - samples.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / double(samples.rows() - 1);

    //   std::cerr << cov.diagonal() << std::endl;
    return cov.diagonal().transpose();
}
*/
