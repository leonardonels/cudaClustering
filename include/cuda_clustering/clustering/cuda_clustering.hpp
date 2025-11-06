#pragma once 

#include <visualization_msgs/msg/marker_array.hpp>

#include "cuda_runtime.h"
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

class cudaExtractCluster
{
  public:
    cudaExtractCluster(cudaStream_t stream = 0);
    ~cudaExtractCluster(void);
    int set(extractClusterParam_t param);
    int extract(float *cloud_in, int nCount, float *output, unsigned int *index);
  private:
    void *m_handle = NULL;
};

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

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
    float clusterMaxX = 0.5, clusterMaxY = 0.5, clusterMaxZ = 0.5, maxHeight = 1.0 ;
    unsigned int memoryAllocated = 0;

    // float *inputEC = NULL;

    // float *outputEC = NULL;

    unsigned int *indexEC = NULL;
    
    extractClusterParam_t ecp;
    cudaStream_t stream = NULL;

    void reallocateMemory(unsigned int sizeEC);

  public:
    CudaClustering(clustering_parameters& param);
    void getInfo();

    void extractClusters(float* input, unsigned int inputSize, float* outputEC, std::shared_ptr<visualization_msgs::msg::Marker> cones);
    ~CudaClustering();
};