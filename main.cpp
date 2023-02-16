// test.c
#include <stdio.h>
#include "Likelihood.h"

int main() {

    double array[5] = {1, 1, 1, 1, 1};
    double d_array[5] = { 0.0 };
    int size = 5;

    printf("likelihood=%f\n", Likelihood::likelihood(array, size));
    printf("%f\n", Likelihood::gradient(array, d_array, size));

  //  dsumarray(array, d_array, size);
    printf("grad=%f,%f,%f,%f,%f", d_array[0], d_array[1], d_array[2], d_array[3], d_array[4]);
}
