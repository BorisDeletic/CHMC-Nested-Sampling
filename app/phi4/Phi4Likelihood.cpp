#include "Phi4Likelihood.h"
#include <unsupported/Eigen/FFT>
#include <iostream>

const Eigen::VectorXd Phi4Likelihood::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * priorWidth - priorWidth / 2;
}

// return the field value at any i,j even if out of bounds.
// fixed boundary of phi=0 outside of lattice
double Phi4Likelihood::FixedBoundaryConditions(const Eigen::VectorXd &theta, int i, int j) {
    if ((i < 0) || (j < 0) || (i > n-1) || (j > n-1)) {
        return 0;
    }

    return theta[i * n + j];
}


const double Phi4Likelihood::LogLikelihood(const Eigen::VectorXd &theta) {
    double fieldAction = 0.0;

    // kinetic term
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fieldAction -= 2 * mKappa * theta[i * n + j] * NeighbourSum(theta, i, j);
        }
    }

    // potential term
    for (int i = 0; i < theta.size(); i++) {
        fieldAction += Potential(theta[i]);
    }

    //lagrangian becomes to T + V after wick rotation

    //lambda=inf gives V(|1|)=0 else inf, so only phi=|1| has non infinite action (non-zero prob)
    //therefore we recover ising model with kappa = 1/T

    return -fieldAction;
}


const Eigen::VectorXd Phi4Likelihood::Gradient(const Eigen::VectorXd &theta) {

  //  Eigen::VectorXd grad = 4 * mKappa * theta;

    // gradient of potential term.
    Eigen::VectorXd grad = 4 * mLambda * theta.cwisePow(3) + (2 - 4 * mLambda) * theta;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grad[i * n + j] -= 4 * mKappa * NeighbourSum(theta, i, j);
        }
    }

    return -grad;
}



double Phi4Likelihood::Potential(double field)
{
// lambda = 1.5 is appropriate for some phase transition properties

//    double V = mLambda * pow(field * field - 1, 2) + field * field;
    double V = mLambda * pow(field, 4) + (1 - 2 * mLambda) * pow(field, 2);

    return V;
}


double Phi4Likelihood::NeighbourSum(const Eigen::VectorXd &theta, int i, int j) {
    double sum = 0.0;

//    int idx_left = j + 1 == n   ? i * n         : i * n + (j+1);
//    int idx_right = j - 1 < 0   ? i * n + (n-1) : i * n + (j-1);
//    int idx_up = i - 1 < 0      ? (n-1) * n + j : (i-1) * n + j;
//    int idx_down = i + 1 == n   ? j             : (i+1) * n + j; // with torus b.c.

//    sum += theta[idx_left];
//    sum += theta[idx_right];
//    sum += theta[idx_up];
//    sum += theta[idx_down];

    sum += FixedBoundaryConditions(theta, i + 1, j);
    sum += FixedBoundaryConditions(theta, i - 1, j);
    sum += FixedBoundaryConditions(theta, i, j + 1);
    sum += FixedBoundaryConditions(theta, i, j - 1);

    return sum;
}

const Eigen::VectorXd Phi4Likelihood::DerivedParams(const Eigen::VectorXd &theta)
{
    const int numDerived = ParamNames().size();
    Eigen::VectorXd derived(numDerived);

    Eigen::VectorXd correlations = SpatialCorrelationFFT(theta);
    for (int i = 0; i < correlations.size(); i++) {
        derived[i] = correlations[i];
    }

    // set last param to magnetisation.
    derived[numDerived - 1] = theta.mean();

    return derived;
}

const std::vector<std::string> Phi4Likelihood::ParamNames() {
    std::vector<std::string> names;

    for (int r = 0; r < n; r++) {
        std::ostringstream corr_name;
        corr_name << "c_" << r;

        names.push_back(corr_name.str());
    }

    names.push_back("mag");
    return names;
}


const Eigen::VectorXd Phi4Likelihood::SpatialCorrelation(const Eigen::VectorXd &theta) {
    auto inbound = [&](int i) {
        return i < n ? i : (n + i) % n;
    }; // only check positive bounds as we count correlation for i+r only

    int maxR = n/2 - 1;
    Eigen::VectorXd correlations = Eigen::VectorXd::Zero(maxR);

    for (int r = 0; r < maxR; r++) {
        //take correlations along diagonal of lattice
        //C(r) = s_ii s_ij + s_ii s_ji   , j = i + r
        for (int i = 0; i < n; i++) {
            int idx0  = i * n + i;
            int idx_h = i * n + inbound(i + r);
            int idx_v = inbound(i + r) * n + i;

            correlations[r] += theta[idx0] * (theta[idx_h] + theta[idx_v]);
        }

        correlations[r] /= 2 * n;
    }

    return correlations;
}


const Eigen::VectorXd Phi4Likelihood::SpatialCorrelationFFT(const Eigen::VectorXd &theta) {
    Eigen::VectorXd correlations = Eigen::VectorXd::Zero(n);

    Eigen::FFT<double> fft;

    Eigen::VectorXd spins(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) {
            spins[j] = theta[i * n + j];
        }

        Eigen::VectorXd rowCorrelation(n);
        Eigen::VectorXcd spinFT(n);

        // Apply convolution theorem to calculate correlation function C(j) for row i
        fft.fwd(spinFT, spins);

        spinFT = spinFT.cwiseAbs2();

        fft.inv(rowCorrelation,spinFT);
        //

        correlations += rowCorrelation;
    }

    correlations /= n; // normalise correlations.

    return correlations;
}




