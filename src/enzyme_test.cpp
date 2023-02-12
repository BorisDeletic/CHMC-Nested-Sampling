#include "enzyme_test.h"

int enzyme_dup;
int enzyme_const;

extern double __enzyme_autodiff(...);

double sumarray(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i] * x[i];
    }
    return sum;
}

double dsumarray(double* x, double* d_x, int size) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff(sumarray,
                             enzyme_dup, x, d_x,
                             enzyme_const, size);
}


