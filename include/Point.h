#ifndef CHMC_NESTED_SAMPLING_POINT_H
#define CHMC_NESTED_SAMPLING_POINT_H

// Markov Chain point
typedef struct MCPoint {
    double* theta;
    int size;
    double likelihood;
} MCPoint;


#endif //CHMC_NESTED_SAMPLING_POINT_H
