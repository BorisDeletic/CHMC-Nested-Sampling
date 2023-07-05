#ifndef CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_ILIKELIHOOD_H

#include <Eigen/Dense>
#include <vector>

class ILikelihood {
public:
    virtual const double LogLikelihood(const Eigen::VectorXd& theta) = 0;
    virtual const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) = 0;

    virtual const Eigen::VectorXd DerivedParams(const Eigen::VectorXd& theta) { return {}; };
    virtual const std::vector<std::string> ParamNames() { return {}; };

    virtual const int GetDimension() = 0;
};

#endif //CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
