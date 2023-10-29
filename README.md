# CHMC-Nested-Sampling
Constrained Hamiltonian Monte Carlo (CHMC) for Nested Sampling (NS)

This project was the focus of my Physics Part III thesis. 
You can read the full report which describes the details how the algorithm works
[here](https://github.com/BorisDeletic/CHMC-Nested-Sampling/blob/main/boris_deletic_report.pdf).

## Getting Started

```angular2html
git clone https://github.com/BorisDeletic/CHMC-Nested-Sampling.git
git submodule update --init --recursive
```

## Compiling

To build one target

```angular2html
mkdir cmake-build-release && cd cmake-build-release
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target GaussianTest
```

or to build all binaries replace last line with 
```angular2html
ninja
```