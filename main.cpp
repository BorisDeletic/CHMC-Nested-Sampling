// test.c
#include <stdio.h>
#include "enzyme_test.h"
#include "Likelihood.h"


int main() {
    Likelihood l;

    double   array[5] = {1, 1, 1, 1, 1};
    double d_array[5] = { 0.0 };
    int size = 5;

    printf("sumarray=%f\n", l.likelihood(array, size));
    l.gradient(array, d_array, size);
    printf("grad=%f,%f,%f,%f,%f", d_array[0], d_array[1], d_array[2], d_array[3], d_array[4]);
}
