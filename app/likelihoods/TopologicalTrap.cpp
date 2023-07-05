#include "TopologicalTrap.h"


TopologicalTrap::TopologicalTrap(int dim, double a, double mu)
    :
    mDim(dim),
    a(a),
    mu(mu)
{
}

const double TopologicalTrap::LogLikelihood(const Eigen::VectorXd &x) {
    // Quadratic term (trap)
    double likelihood = -x.cwisePow(2).sum();

    // Gaussian term (global max)
    likelihood += a * exp( -((x.array() - mu) / var).pow(2).sum() );

    return likelihood;
}

const Eigen::VectorXd TopologicalTrap::Gradient(const Eigen::VectorXd &x) {
    // Quadratic term
    Eigen::VectorXd grad = - 2 * x;

    grad = grad - 2 * a * exp( -((x.array() - mu) / var).pow(2).sum() ) * (x.array() - mu).matrix() / (var * var);

    return grad;
}
