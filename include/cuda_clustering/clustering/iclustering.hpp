#pragma once 
#include "cuda_clustering/clustering/cluster_filtering/icluster_filtering.hpp"

class IClustering 
{
    public:
        virtual void extractClusters(float* input, unsigned int inputSize, float* outputEC, std::shared_ptr<visualization_msgs::msg::Marker> cones) = 0;
        virtual void getInfo() = 0;
        IClusterFiltering* filter;
};