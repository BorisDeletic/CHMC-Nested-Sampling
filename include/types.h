#ifndef CHMC_NESTED_SAMPLING_TYPES_H
#define CHMC_NESTED_SAMPLING_TYPES_H

#include <Eigen/Dense>

// Markov Chain point
typedef struct MCPoint {
    const Eigen::VectorXd theta;
    const Eigen::VectorXd derived;
    const double likelihood;
    const double birthLikelihood;
    const int reflections = 0;
    const int steps = 0;
    const double acceptProbability = 1;
    bool rejected = false;
} MCPoint;


inline bool operator<(const MCPoint& a, const MCPoint& b) {
    return a.likelihood < b.likelihood;
}


typedef struct NSConfig {
    int numLive;
    int maxIters;
    double precisionCriterion;
    double reflectionRateThreshold;
    bool logDiagnostics;
} NSConfig;


typedef struct NSInfo {
    int iter;
    int numLive;
    double reflectRateThreshold;
    double logZ;
    double logZLive;
} NSSummary;


#endif //CHMC_NESTED_SAMPLING_TYPES_H
