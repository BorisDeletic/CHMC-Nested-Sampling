#ifndef CHMC_NESTED_SAMPLING_PHI4LFT_H
#define CHMC_NESTED_SAMPLING_PHI4LFT_H

#include <string>
#include <vector>
#include <Eigen/Dense>

typedef struct Posterior {
    std::vector<double> posteriorWeights;
    std::vector<Eigen::VectorXd> derivedParams;
} Posterior;


void runPhi4(std::string fname, double kappa, double lambda);

const Posterior ReadPosteriorFile(std::string fname);
double calculateMeanMag(const Posterior&);

void generatePhaseDiagramData();
void posteriorAnalysis();

#endif //CHMC_NESTED_SAMPLING_PHI4LFT_H
