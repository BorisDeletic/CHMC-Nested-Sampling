#include "Eggbox.h"

Eggbox::Eggbox(int dim) : mDim(dim) {
}

const double Eggbox::LogLikelihood(const Eigen::VectorXd &theta) {

    const Eigen::VectorXd t = 2.0 * tmax * (theta.array() - 0.5);

    return pow(2.0 + cos(t[0] / 2.0) * cos(t[1] / 2.0), 5.0);
}

const Eigen::VectorXd Eggbox::Gradient(const Eigen::VectorXd &theta) {
    double x = 2.0 * tmax * (theta[0] - 0.5);
    double y = 2.0 * tmax * (theta[1] - 0.5);

    Eigen::Vector2d grad = {
            -5 * tmax * sin(x / 2.0) * cos(y / 2) * pow(2 + cos(x/2) * cos(y/2), 4),
            -5 * tmax * cos(x / 2.0) * sin(y / 2) * pow(2 + cos(x/2) * cos(y/2), 4)
    };

    return -grad;
}


