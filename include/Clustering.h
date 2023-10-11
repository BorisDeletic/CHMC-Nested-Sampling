#ifndef CHMC_NESTED_SAMPLING_CLUSTERING_H
#define CHMC_NESTED_SAMPLING_CLUSTERING_H

#include "types.h"
#include <set>
#include <vector>

class Clustering {
public:
    Clustering(std::multiset<MCPoint>& points);

    void AssignClusters();

private:
    std::vector<std::vector<int>> FindAllNN();

    std::vector<int> CalculateClusters(std::vector<std::vector<int>>& allNN, int k);
    bool KNeighbours(std::vector<int>& A, std::vector<int>& B, int k);

    std::multiset<MCPoint>& mLivePoints;

};

#endif //CHMC_NESTED_SAMPLING_CLUSTERING_H
