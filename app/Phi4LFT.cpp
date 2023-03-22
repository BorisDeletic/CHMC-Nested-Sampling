#include "Phi4LFT.h"

const Eigen::VectorXd Phi4Likelihood::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * priorWidth - priorWidth / 2;
}


const double Phi4Likelihood::LogLikelihood(const Eigen::VectorXd &theta) {
    double fieldAction = 0.0;

    // kinetic term
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fieldAction += mKappa * Laplacian(theta, i, j);
        }
    }

    // potential term
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fieldAction += Potential(theta[i * n + j]);
        }
    }

    //lagrangian becomes to T + V after wick rotation

    //lambda=inf gives V(|1|)=0 else inf, so only phi=|1| has non infinite action (non-zero prob)
    //therefore we recover ising model with kappa = 1/T

    return -fieldAction;
}


const Eigen::VectorXd Phi4Likelihood::Gradient(const Eigen::VectorXd &theta) {
    Eigen::VectorXd grad = (4 * mKappa - 4 * mLambda + 2) * theta;

    grad += 4 * mLambda * theta.cwisePow(3);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grad[i * n + j] -= mKappa * NeighbourSum(theta, i, j);
        }
    }

    return -grad;
}



double Phi4Likelihood::Potential(double field)
{
// lambda = 1.5 is appropriate for some phase transition properties

    double V = mLambda * pow(field * field - 1, 2) + field * field;
  //  double V = mLambda * pow(field * field - 1, 2);

    return V;
}


double Phi4Likelihood::Laplacian(const Eigen::VectorXd& theta, int i, int j)
{
    double kinetic = 0.0;

    int idx = i * n + j;

    int idx_left = j + 1 == n   ? i * n         : i * n + (j+1);
    int idx_right = j - 1 < 0   ? i * n + (n-1) : i * n + (j-1);
    int idx_up = i - 1 < 0      ? (n-1) * n + j : (i-1) * n + j;
    int idx_down = i + 1 == n   ? j             : (i+1) * n + j; // with torus b.c.

    kinetic += 2 * pow(theta[idx], 2);
    kinetic -= theta[idx] * theta[idx_left];
    kinetic -= theta[idx] * theta[idx_right];
    kinetic -= theta[idx] * theta[idx_up];
    kinetic -= theta[idx] * theta[idx_down];

    return kinetic;
}

double Phi4Likelihood::NeighbourSum(const Eigen::VectorXd &theta, int i, int j) {
    double sum = 0.0;

    int idx_left = j + 1 == n   ? i * n         : i * n + (j+1);
    int idx_right = j - 1 < 0   ? i * n + (n-1) : i * n + (j-1);
    int idx_up = i - 1 < 0      ? (n-1) * n + j : (i-1) * n + j;
    int idx_down = i + 1 == n   ? j             : (i+1) * n + j; // with torus b.c.

    sum += theta[idx_left];
    sum += theta[idx_right];
    sum += theta[idx_up];
    sum += theta[idx_down];

    return sum;
}


