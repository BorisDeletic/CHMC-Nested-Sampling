//
// Created by Boris Deletic on 14/02/2023.
//

#ifndef CHMC_NESTED_SAMPLING_POINT_H
#define CHMC_NESTED_SAMPLING_POINT_H

struct Point {
    double* theta;
    int size;
    double likelihood;
};

#endif //CHMC_NESTED_SAMPLING_POINT_H
