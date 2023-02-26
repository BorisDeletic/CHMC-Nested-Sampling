#ifndef CHMC_NESTED_SAMPLING_TYPES_H
#define CHMC_NESTED_SAMPLING_TYPES_H

#include <Eigen/Dense>

// Markov Chain point
typedef struct MCPoint {
    const Eigen::VectorXd theta;
    const double likelihood;
} MCPoint;

bool operator<(const MCPoint& a, const MCPoint& b) {
    return a.likelihood < b.likelihood;
}


#endif //CHMC_NESTED_SAMPLING_TYPES_H
