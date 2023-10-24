#include "Shells.h"
#include <iostream>

Shells::Shells(int dim, double radius, double width, double center)
    :
    mDim(dim),
    r(radius),
    w(width),
    c(center)
{
}

double Shells::LogCircle(const Eigen::VectorXd &theta, double center) {
    double prefac = log(1.0 / sqrt(2 * M_PI * w * w));

    Eigen::VectorXd center_vec = Eigen::VectorXd::Zero(mDim);
    center_vec[0] = center;

    double d = (theta - center_vec).norm();

    return prefac - pow(d - r, 2) / (2 * w * w);
}


const double Shells::LogLikelihood(const Eigen::VectorXd &theta) {
    return LogAdd(LogCircle(theta, c), LogCircle(theta, -c));
}


const Eigen::VectorXd Shells::Gradient(const Eigen::VectorXd &theta) {
 //   double prefac = 1 / (exp(LogCircle(theta, c)) + exp(LogCircle(theta, -c)));

    Eigen::VectorXd grad = (GradientCircle(theta, c) + GradientCircle(theta, -c));


    return grad;
}


const Eigen::VectorXd Shells::GradientCircle(const Eigen::VectorXd &theta, double center) {
//    double prefac = 1 / sqrt(2 * M_PI * w * w);

    Eigen::VectorXd center_vec = Eigen::VectorXd::Zero(mDim);
    center_vec[0] = center;

    double d = (theta - center_vec).norm();

    Eigen::VectorXd grad = -(theta - center_vec) * (d - r) / d * exp(LogCircle(theta, center));

    return grad;
}

double Shells::LogAdd(double logA, double logB) {
    if (logA > logB) {
        return logA + log(1 + exp(logB - logA));
    } else {
        return logB + log(1 + exp(logA - logB));
    }
}
