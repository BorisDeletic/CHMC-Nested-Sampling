#include "PosteriorAnalysis.h"
#include <fstream>
#include <iostream>

//std::string phase_dir = "phase_diagram";

const double kappaMax = 1;
const int resolution = 10;

const Posterior ReadPosteriorFile(std::string fname) {
    std::ifstream posteriorFile;
    posteriorFile.open(fname);

    std::vector<double> posteriorWeights;
    std::vector<Eigen::VectorXd> derivedParams;

    if (posteriorFile.is_open()) {
        std::string line, str_vals;

        while (std::getline(posteriorFile,line)) {
            std::stringstream ssline(line);
            std::vector<double> row;

            while(std::getline(ssline, str_vals, ' '))
            {
                double val = std::stod(str_vals);
                row.push_back(val);
                std::cout << val << std::endl;
            }


        }
        posteriorFile.close();
    }

    return Posterior();
}


double calculateMeanMag(const Posterior& posterior) {
    int numSamples = posterior.posteriorWeights.size();
    Eigen::VectorXd absMag(numSamples);

    for (int i = 0; i < numSamples; i++) {
        // scale observable by posterior weight
        absMag[i] = posterior.posteriorWeights[i] * abs(posterior.derivedParams[i][0]);
    }

    double meanMag = absMag.mean();

    return meanMag;
}


void posteriorAnalysis() {
    std::ofstream magFile;
    magFile.open("mags.csv");

    magFile << "kappa,lambda,mag" << std::endl;

    for (double k = 0; k < kappaMax; k += kappaMax / resolution) {
        std::ostringstream fname;
       // fname << phase_dir;
      //  fname << "/Phi4_" << std::setprecision(5) << std::fixed << k << "_" << l << ".posterior";

        const Posterior posteriorData = ReadPosteriorFile(fname.str());
        const double meanMag = calculateMeanMag(posteriorData);

      //  magFile << k << "," << l << "," << meanMag << std::endl;
    }

  //  magFile.close();
}
