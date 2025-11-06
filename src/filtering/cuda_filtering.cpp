#include "cuda_clustering/filtering/cuda_filtering.hpp"

CudaFilter::CudaFilter(float upFilterLimits, float downFilterLimits)
{
  FilterType_t type = PASSTHROUGH;

  this->setP.type = type;
  this->setP.dim = 2;
  this->setP.upFilterLimits = upFilterLimits;
  this->setP.downFilterLimits = downFilterLimits;
  this->setP.limitsNegative = false;

  cudaStreamCreate ( &stream );
};

void CudaFilter::filterPoints(float* inputData, unsigned int inputSize, float** output, unsigned int* outputSize)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  cudaFilter filterTest(stream);
  std::cout << "\n------------ CUDA PassThrough ---------------- "<< std::endl;

  filterTest.set(this->setP);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  
  filterTest.filter(*output, outputSize, inputData, inputSize);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough Time: %f ms.", time_span.count());
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough before filtering: %d", inputSize);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough after filtering: %d", *outputSize);
}