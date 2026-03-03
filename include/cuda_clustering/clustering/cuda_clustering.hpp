#pragma once 

#include <visualization_msgs/msg/marker_array.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

#include "cuda_clustering/clustering/iclustering.hpp"

typedef struct {
  unsigned int minClusterSize;
  unsigned int maxClusterSize;
  float voxelX;
  float voxelY;
  float voxelZ;
  int countThreshold;
} extractClusterParam_t;

struct clustering_parameters
{
    struct clustering
    {
        float voxelX, voxelY, voxelZ;
        unsigned int countThreshold, minClusterSize, maxClusterSize;
    } clustering;

    cluster_filter filtering;
};


class CudaClustering : public IClustering
{
  private:
    extractClusterParam_t ecp;
    cudaStream_t stream = NULL;

    // -------------------------------------------------------------------------
    // Device vectors replace cudaMallocManaged
    // -------------------------------------------------------------------------
    // voxelization
    thrust::device_vector<int>          d_voxelKeys;     // voxel hash per point
    thrust::device_vector<int>          d_sortedKeys;    // sorted voxel hashes
    thrust::device_vector<unsigned int> d_sortedIndices; // original indices after sort
    thrust::device_vector<int>          d_uniqueKeys;    // unique voxel hashes
    thrust::device_vector<unsigned int> d_voxelCounts;   // points per voxel
    thrust::device_vector<unsigned int> d_voxelOffsets;  // start offset per voxel

    // union-find clustering
    thrust::device_vector<int>          d_parent;        // union-find parent array
    thrust::device_vector<int>          d_clusterLabels; // final cluster label per voxel

    // output
    thrust::device_vector<unsigned int> d_indexEC;       // cluster index array
    thrust::host_vector<unsigned int>   h_indexEC;       // host mirror

    double totalTime = 0.0;
    unsigned int iterations = 0;

  public:
    CudaClustering(clustering_parameters& param);
    ~CudaClustering();
    void getInfo();

    void extractClusters(float* input, unsigned int inputSize, float* outputEC,
                         std::shared_ptr<visualization_msgs::msg::Marker> cones);
};