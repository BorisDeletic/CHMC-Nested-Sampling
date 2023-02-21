#ifndef CHMC_NESTED_SAMPLING_TYPES_H
#define CHMC_NESTED_SAMPLING_TYPES_H

#include <Eigen/Dense>

// Markov Chain point
typedef struct MCPoint {
    Eigen::VectorXd theta;
    double likelihood;
} MCPoint;


#endif //CHMC_NESTED_SAMPLING_TYPES_H
