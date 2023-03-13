#include "CHMC.h"
#include "Hamiltonian.h"
#include <memory>
#include <iostream>

CHMC::CHMC(ILikelihood& likelihood, double epsilon, int pathLength)
    :
    mLikelihood(likelihood),
    mPathLength(pathLength),
    gen(rd()),
    mNorm(0, 1),
    mUniform(0, 1)
{
    mHamiltonian = std::make_unique<Hamiltonian>(likelihood, epsilon);
}

CHMC::~CHMC() = default;


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
        MCPoint next = SamplePoint(last, -1e30);
        samples.row(i) = next.theta;
    }

    Eigen::VectorXd newMetric = CalculateVar(samples).cwiseInverse();
   // std::cerr << std::endl << newMetric << std::endl;

    mHamiltonian->SetMetric(newMetric);
    return true;
}


Eigen::VectorXd CHMC::SampleP(const int size) {
    Eigen::VectorXd p(size);

    Eigen::VectorXd sqrtMetric = mHamiltonian->GetMetric().cwiseSqrt();

    for (int i = 0; i < size; i++) {
        p(i) = mNorm(gen) * sqrtMetric(i);
    }

    return p;
}


const MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    const Eigen::VectorXd p = SampleP(old.theta.size());

    mHamiltonian->SetHamiltonian(old.theta, p, likelihoodConstraint);
    const double initEnergy = mHamiltonian->GetEnergy();

    for (int i = 0; i < mPathLength; i++) {
        mHamiltonian->Evolve();
    }

    const double acceptProb = exp(initEnergy - mHamiltonian->GetEnergy());
    const double r = mUniform(gen);
    if (acceptProb > r)
    {
        MCPoint newPoint = {
                mHamiltonian->GetX(),
                mHamiltonian->GetLikelihood(),
                likelihoodConstraint
        };
        return newPoint;

    } else
    {
     //   std::cerr << "REJECT POINT";
        return old;
    }
}

Eigen::VectorXd CHMC::CalculateVar(const Eigen::MatrixXd& samples) {
    Eigen::MatrixXd centered = samples.rowwise() - samples.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / double(samples.rows() - 1);

 //   std::cerr << cov.diagonal() << std::endl;
    return cov.diagonal().transpose();
}

const Eigen::VectorXd& CHMC::GetMetric() {
    return mHamiltonian->GetMetric();
}

