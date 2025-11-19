#include "cuda_clustering/filtering/cuda_filtering.hpp"

CudaFilter::CudaFilter(float upX, float downX, float upY, float downY, float upZ, float downZ)
{
  FilterType_t type = PASSTHROUGH;

  this->setP.type = type;
  this->setP.limitsNegative = false;
  
  // Store filter limits for each axis
  this->upLimitX = upX;
  this->downLimitX = downX;
  this->upLimitY = upY;
  this->downLimitY = downY;
  this->upLimitZ = upZ;
  this->downLimitZ = downZ;
  
  // Determine which axes to filter (if limits are different)
  this->filterX = (upX != 1e10 && downX != -1e10);
  this->filterY = (upY != 1e10 && downY != -1e10);
  this->filterZ = (upZ != 1e10 && downZ != -1e10);

  cudaStreamCreate ( &stream );
};

void CudaFilter::filterPoints(float* inputData, unsigned int inputSize, float** output, unsigned int* outputSize)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::cout << "\n------------ CUDA PassThrough (XYZ) ---------------- "<< std::endl;
  
  unsigned int initialSize = inputSize;
  float* currentInput = inputData;
  float* currentOutput = *output;
  unsigned int currentSize = inputSize;
  
  // Reallocate temp buffer if needed
  if(memoryAllocated < inputSize * 4 * sizeof(float)){
    if(tempOutput != nullptr) cudaFree(tempOutput);
    cudaMallocManaged(&tempOutput, sizeof(float) * 4 * inputSize);
    memoryAllocated = inputSize * 4 * sizeof(float);
  }
  
  cudaFilter filterTest(stream);
  
  // Filter X axis (dim=0)
  if(this->filterX) {
    unsigned int prevSize = currentSize;
    this->setP.dim = 0;
    this->setP.upFilterLimits = this->upLimitX;
    this->setP.downFilterLimits = this->downLimitX;
    filterTest.set(this->setP);
    cudaStreamSynchronize(stream);
    
    filterTest.filter(tempOutput, &currentSize, currentInput, currentSize);
    cudaStreamSynchronize(stream);
    RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "X filter: %d -> %d points", prevSize, currentSize);
    currentInput = tempOutput;
  }
  
  // Filter Y axis (dim=1)
  if(this->filterY) {
    unsigned int prevSize = currentSize;
    this->setP.dim = 1;
    this->setP.upFilterLimits = this->upLimitY;
    this->setP.downFilterLimits = this->downLimitY;
    filterTest.set(this->setP);
    cudaStreamSynchronize(stream);
    
    filterTest.filter(currentOutput, &currentSize, currentInput, currentSize);
    cudaStreamSynchronize(stream);
    RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "Y filter: %d -> %d points", prevSize, currentSize);
    currentInput = currentOutput;
  }
  
  // Filter Z axis (dim=2)
  if(this->filterZ) {
    unsigned int prevSize = currentSize;
    this->setP.dim = 2;
    this->setP.upFilterLimits = this->upLimitZ;
    this->setP.downFilterLimits = this->downLimitZ;
    filterTest.set(this->setP);
    cudaStreamSynchronize(stream);
    
    // If Y wasn't filtered, output directly; otherwise use tempOutput
    float* finalOutput = this->filterY ? tempOutput : currentOutput;
    filterTest.filter(finalOutput, &currentSize, currentInput, currentSize);
    cudaStreamSynchronize(stream);
    RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "Z filter: %d -> %d points", prevSize, currentSize);
    
    // Copy to final output if needed
    if(finalOutput != currentOutput) {
      cudaMemcpy(currentOutput, finalOutput, currentSize * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
      cudaStreamSynchronize(stream);
    }
  }
  
  *outputSize = currentSize;
  cudaDeviceSynchronize();
  
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough Time: %f ms.", time_span.count());
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough: %d -> %d points (filtered on X:%s Y:%s Z:%s)", 
              initialSize, *outputSize,
              this->filterX ? "yes" : "no",
              this->filterY ? "yes" : "no",
              this->filterZ ? "yes" : "no");
}

CudaFilter::~CudaFilter() {
  if(tempOutput != nullptr) {
    cudaFree(tempOutput);
    tempOutput = nullptr;
  }
  if(stream != NULL) {
    cudaStreamDestroy(stream);
  }
}