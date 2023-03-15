#include "Phi4LFT.h"

const Eigen::VectorXd Phi4Likelihood::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * priorWidth - priorWidth / 2;
}

const double Phi4Likelihood::LogLikelihood(const Eigen::VectorXd &theta) {

}

const Eigen::VectorXd Phi4Likelihood::Gradient(const Eigen::VectorXd &theta) {
    return Eigen::VectorXd();
}
