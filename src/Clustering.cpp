
#include "Clustering.h"
#include <algorithm>
#include <iostream>


Clustering::Clustering(std::multiset<MCPoint> &points)
    :
    mLivePoints(points)
{

}

void Clustering::AssignClusters() {
    int numLive = mLivePoints.size();

    std::vector<int> clusters(numLive, -1);

    std::vector<std::vector<int>> allNN = FindAllNN();

    for (int k = 2; k < numLive; k++) {
        std::vector<int> newClusters = CalculateClusters(allNN, k);

        // no changes in clusters means we are finished
        if (clusters == newClusters)
            break;

        clusters = newClusters;
    }

    // repeat on sub clusters

    //write clusters to live points
    auto pointIt = mLivePoints.begin();
    for (int i = 0; i < numLive; i++) {
        MCPoint point = *pointIt;
        point.cluster = clusters[i];

        pointIt++;

        std::cout << i << ": " << clusters[i] << std::endl;
    }
}


std::vector<int> Clustering::CalculateClusters(std::vector<std::vector<int>>& allNN, int k) {
    int numLive = mLivePoints.size();

    std::vector<int> clusters(numLive);
    for (int i = 0; i < clusters.size(); i++) {
        clusters[i] = i;
    }

    for (int i = 0; i < clusters.size(); i++)
    {
        for (int j = i + 1; j < clusters.size(); j++) {
            if (clusters[i] == clusters[j]) continue;

            if (KNeighbours(allNN[i], allNN[j], k)) {
                clusters[i] = std::min(clusters[i], clusters[j]);
                clusters[j] = std::min(clusters[i], clusters[j]);
            }
        }
    }

    return clusters;
}



// index i holds a list of all nearest neighbours to point i
std::vector<std::vector<int>> Clustering::FindAllNN() {
    int numLive = mLivePoints.size();

    std::vector<std::vector<int>> allNN(numLive, std::vector<int>(numLive));
    std::vector<std::vector<double>> allDistances(numLive);

    for (int i = 0; i < allDistances.size(); i++) {
        auto point_i = mLivePoints.begin();
        std::advance(point_i, i);

        const Eigen::VectorXd& pos_i = point_i->theta;

        for (int j = 0; j < allDistances.size(); j++) {
            auto point_j = mLivePoints.begin();
            std::advance(point_j, j);
            const Eigen::VectorXd& pos_j = point_j->theta;

            double distance = (pos_i - pos_j).norm();

            allDistances[i].push_back(distance);
        }
    }

    for (int i = 0; i < allNN.size(); i++)
    {
        std::sort(std::begin(allNN[i]), std::end(allNN[i]),
                  [&allDistances, i](const auto & lhs, const auto & rhs)
                  {
                      return allDistances[i][lhs] < allDistances[i][rhs];
                  }
        );
    }

    return allNN;
}


// Are A & B mutually within k nearest neighbours
bool Clustering::KNeighbours(std::vector<int> &A, std::vector<int> &B, int k) {
    if (k >= A.size()) return true;

    // Search for point B (given by B[0]) in first k neighbours of A
    bool ANN = std::find(A.begin(), A.begin() + k, B[0]) != A.begin() + k;
    bool BNN = std::find(B.begin(), B.begin() + k, A[0]) != B.begin() + k;

    if (ANN && BNN)
        return true;
    else
       return false;
}



