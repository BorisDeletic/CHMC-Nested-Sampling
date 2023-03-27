#include "Adapter.h"

Adapter::Adapter(CHMC& chmc, double initEpsilon, int initPathLength, const Eigen::VectorXd metric)
    :
    mCHMC(chmc),
    mEpsilon(initEpsilon),
    mPathLength(initPathLength),
    mMetric(metric)
{
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