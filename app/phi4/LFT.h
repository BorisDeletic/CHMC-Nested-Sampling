#ifndef CHMC_NESTED_SAMPLING_LFT_H
#define CHMC_NESTED_SAMPLING_LFT_H

#include <string>
#include <vector>
#include <Eigen/Dense>


void runPhi4(std::string fname, int n, double kappa, double lambda);

void generatePhaseDiagramData();

#endif //CHMC_NESTED_SAMPLING_LFT_H
