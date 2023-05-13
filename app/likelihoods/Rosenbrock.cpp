#include "Rosenbrock.h"
#include "Likelihood.h"

const double RosenbrockLikelihood::LogLikelihood(const Eigen::VectorXd& theta)
{
    double* data = const_cast<double *>(theta.data());
    double logL = Likelihood::likelihood(data, mDim, A);

    return logL;
};


const Eigen::VectorXd RosenbrockLikelihood::Gradient(const Eigen::VectorXd& theta)
{
    double* d_theta = (double*) malloc(mDim * sizeof(double));
    memset(d_theta, 0, mDim * sizeof(double));

    double* data = const_cast<double *>(theta.data());

    Likelihood::gradient(data, d_theta, mDim, A);

    Eigen::VectorXd grad(mDim);

    for (int i = 0; i < mDim; i++) {
        grad[i] = d_theta[i];
    }

    free(d_theta);

    return grad;
};


const Eigen::VectorXd RosenbrockLikelihood::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * priorWidth - priorWidth / 2;
}