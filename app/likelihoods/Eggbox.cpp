#include "Eggbox.h"

Eggbox::Eggbox(int dim) : mDim(dim) {
}

const double Eggbox::LogLikelihood(const Eigen::VectorXd &theta) {

    const Eigen::VectorXd t = 2.0 * tmax * (theta.array() - 0.5);

    double cos_term = 1;
    for (auto x : t) {
        cos_term *= cos(x / 2);
    }

    return pow(2.0 + cos_term, 5.0);
}

const Eigen::VectorXd Eggbox::Gradient(const Eigen::VectorXd &theta) {
    Eigen::ArrayXd t = 2.0 * tmax * (theta.array() - 0.5);

    Eigen::ArrayXd sin_x = sin(t / 2);
    Eigen::ArrayXd cos_x = cos(t / 2);

    double cos_term = 1;
    for (auto x : t) {
        cos_term *= cos(x / 2);
    }

    // have to 'replace' cos(x) with sin(x) from the cos term
    Eigen::VectorXd grad = -5 * tmax * sin_x / cos_x * cos_term;
    grad *= pow(2 + cos_term, 4);

    // dL/dx = -5 * tmax * sin(x / 2.0) * cos(y / 2) * pow(2 + cos(x/2) * cos(y/2), 4),

    return -grad;
}


