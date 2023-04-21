#ifndef CHMC_NESTED_SAMPLING_PHI4ANALYSIS_H
#define CHMC_NESTED_SAMPLING_PHI4ANALYSIS_H

#include <vector>
#include <Eigen/Dense>

typedef struct Posterior {
    std::vector<double> posteriorWeights;
    std::vector<Eigen::VectorXd> derivedParams;
} Posterior;

const Posterior ReadPosteriorFile(std::string fname);
double calculateMeanMag(const Posterior&);
void posteriorAnalysis();


class Phi4Analysis {

};


#endif //CHMC_NESTED_SAMPLING_PHI4ANALYSIS_H
