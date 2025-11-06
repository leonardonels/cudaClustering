#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/clustering/cluster_filtering/dimension_filter.hpp"

CudaClustering::CudaClustering(clustering_parameters& param){
  this->ecp.minClusterSize = param.clustering.minClusterSize;           // Minimum cluster size to filter out noise
  this->ecp.maxClusterSize = param.clustering.maxClusterSize;        // Maximum size for large objects
  this->ecp.voxelX = param.clustering.voxelX;                  // Down-sampling resolution in X (meters)
  this->ecp.voxelY = param.clustering.voxelY;                  // Down-sampling resolution in Y (meters)
  this->ecp.voxelZ = param.clustering.voxelZ;                 // Down-sampling resolution in Z (meters)
  this->ecp.countThreshold = param.clustering.countThreshold;           // Minimum points per voxel

  filter = new DimensionFilter(param.filtering);
  cudaStreamCreate ( &stream );
}

void CudaClustering::getInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
      cudaGetDeviceProperties(&prop, i);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"----device id: %d info----\n", i);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  GPU : %s \n", prop.name);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Capability: %d.%d\n", prop.major, prop.minor);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Const memory: %luKB\n", prop.totalConstMem  >> 10);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  warp size: %d\n", prop.warpSize);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  threads in a block: %d\n", prop.maxThreadsPerBlock);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"\n");
}

void CudaClustering::reallocateMemory(unsigned int sizeEC){
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "REALLOC");
  cudaFree(indexEC);
  cudaStreamSynchronize(stream);
  cudaMallocManaged(&indexEC, sizeof(unsigned int) * 4 * (sizeEC), cudaMemAttachHost);
  cudaStreamSynchronize(stream);
  cudaStreamAttachMemAsync (stream, indexEC);
  cudaStreamSynchronize(stream);
}

void CudaClustering::extractClusters(float* input, unsigned int inputSize, float* outputEC, std::shared_ptr<visualization_msgs::msg::Marker> cones)
{
  std::cout << "\n------------ CUDA Clustering ---------------- "<< std::endl;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  if(memoryAllocated < inputSize){
    reallocateMemory(inputSize);
    memoryAllocated = inputSize;
  }

  // cudaMemcpyAsync(outputEC, input, sizeof(float) * 4 * inputSize, cudaMemcpyHostToDevice, stream);
  
  cudaMemsetAsync(indexEC, 0, sizeof(unsigned int) * 4 * (inputSize), stream);
  cudaStreamSynchronize(stream);
  
  cudaExtractCluster cudaec(stream);
  cudaec.set(this->ecp);
  cudaStreamSynchronize(stream);

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA Memory Time: %f ms.", time_span.count());

  cudaec.extract(input, inputSize, outputEC, indexEC);
  cudaStreamSynchronize(stream);

  for (size_t i = 1; i <= indexEC[0]; i++)
  {
    unsigned int outoff = 0;
    for (size_t w = 1; w < i; w++)
    {
      if (i>1) {
        outoff += indexEC[w];
      }
    }
    std::optional<geometry_msgs::msg::Point> pnt_opt = filter->analiseCluster(&outputEC[outoff*4], indexEC[i]);
    //RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "Possible cluster of %d points: detected as %d.", indexEC[i], pnt_opt.has_value());
    if(pnt_opt.has_value()){
      cones->points.push_back(pnt_opt.value());
    }
  }
  std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t3 - t2);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA extract by Time: %f ms.", time_span.count());
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t3 - t1);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA Total Time: %f ms.", time_span.count());
  /*end*/

  // cudaFree(input);
}
CudaClustering::~CudaClustering(){
  // cudaFree(inputEC);
  // cudaFree(outputEC);
  cudaFree(indexEC);
}
