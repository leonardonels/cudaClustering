#pragma once 

#include <visualization_msgs/msg/marker_array.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
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
    cluster_filter filterParams;
    cudaStream_t stream = NULL;

    // -------------------------------------------------------------------------
    // Device vectors — all intermediate data stays on GPU
    // -------------------------------------------------------------------------
    // bounding box + grid dims (written by kernels, read by kernels)
    thrust::device_vector<float> d_bbox;        // [minX, minY, minZ, maxX, maxY, maxZ]
    thrust::device_vector<int>   d_grid;        // [gridX, gridY]

    // voxelization
    thrust::device_vector<int>          d_voxelKeys;     // voxel hash per point
    thrust::device_vector<int>          d_sortedKeys;    // sorted voxel hashes
    thrust::device_vector<unsigned int> d_sortedIndices; // original indices after sort
    thrust::device_vector<int>          d_uniqueKeys;    // unique voxel hashes
    thrust::device_vector<unsigned int> d_voxelCounts;   // points per voxel
    thrust::device_vector<int>          d_filteredKeys;  // voxels surviving countThreshold

    // union-find clustering
    thrust::device_vector<int>          d_parent;        // union-find parent array

    // per-point cluster labels
    thrust::device_vector<int>          d_pointLabels;   // cluster root label per point

    // per-cluster data
    thrust::device_vector<int>          d_uniqueLabels;  // unique root labels
    thrust::device_vector<int>          d_labelMap;      // root → compact cluster id
    thrust::device_vector<float>        d_clusterBBox;   // [6 * numClusters]
    thrust::device_vector<unsigned int> d_clusterSizes;  // points per cluster

    // output cones
    thrust::device_vector<float>        d_conePoints;    // [3 * maxCones]
    thrust::device_vector<unsigned int> d_numCones;      // single element counter

    double totalTime = 0.0;
    unsigned int iterations = 0;

  public:
    CudaClustering(clustering_parameters& param);
    ~CudaClustering();
    void getInfo();

    void extractClusters(float* input, unsigned int inputSize, float* outputEC,
                         std::shared_ptr<visualization_msgs::msg::Marker> cones);
};