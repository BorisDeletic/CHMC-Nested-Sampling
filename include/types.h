#ifndef CHMC_NESTED_SAMPLING_TYPES_H
#define CHMC_NESTED_SAMPLING_TYPES_H

// Markov Chain point
typedef struct MCPoint {
    const double* theta;
    const int size;
    const double likelihood;
} MCPoint;


#endif //CHMC_NESTED_SAMPLING_TYPES_H
