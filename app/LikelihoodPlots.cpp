#include "LikelihoodPlots.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void generateContours(ILikelihood& likelihood, std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("isocontours.dat");
    file << "x y z" << std::endl;

    double xSpacing = (xran.second - xran.first) / 100;
    double ySpacing = (yran.second - yran.first) / 100;

    for (double i = xran.first; i < xran.second; i += xSpacing) {
        for (double j = yran.first; j < yran.second; j += ySpacing) {
            Eigen::VectorXd theta = Eigen::VectorXd::Zero(likelihood.GetDimension());
            theta[0] = i;
            theta[1] = j;
            const double loglike = likelihood.LogLikelihood(theta);

            file << std::setprecision(3) << std::fixed << i << " " << j << " " <<
                 std::setprecision(4) << std::fixed << loglike << std::endl;
        }
        file << std::endl;
    }

    file.close();
}


void generateGradientField(ILikelihood& likelihood, std::pair<float, float>& xran, std::pair<float, float>& yran) {
    std::ofstream file;
    file.open("gradient.dat");
    file << "x y dx dy" << std::endl;

    double xSpacing = (xran.second - xran.first) / 20;
    double ySpacing = (yran.second - yran.first) / 20;

    for (double i = xran.first; i < xran.second; i += xSpacing) {
        for (double j = yran.first; j < yran.second; j += ySpacing) {
            Eigen::VectorXd theta = Eigen::VectorXd::Zero(likelihood.GetDimension());
            theta[0] = i;
            theta[1] = j;

            Eigen::VectorXd grad = likelihood.Gradient(theta).normalized() / 15;

            file << std::setprecision(3) << std::fixed << i << " " << j << " " <<
                 std::setprecision(4) << std::fixed << grad[0] << " " << grad[1] << std::endl;
        }
        file << std::endl;
    }

    file.close();
}


void generateLikelihoodPlot(ILikelihood& likelihood, std::pair<float, float> xran, std::pair<float, float> yran) {
    generateContours(likelihood, xran, yran);
    generateGradientField(likelihood, xran, yran);
}
