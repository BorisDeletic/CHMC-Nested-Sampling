// test.c
#include <stdio.h>
#include "enzyme_test.h"

int main() {
    for(double i=1; i<5; i++)
        printf("square(%f)=%f, dsquare(%f)=%f\n", i, square(i), i, dsquare(i));
}
