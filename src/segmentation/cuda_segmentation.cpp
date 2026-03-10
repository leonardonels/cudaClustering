#include "cuda_clustering/segmentation/cuda_segmentation.hpp"

CudaSegmentation::CudaSegmentation(segParam_t &params)
{
  segP.distanceThreshold = params.distanceThreshold;
  segP.maxIterations = params.maxIterations;
  segP.probability = params.probability;
  segP.optimizeCoefficients = params.optimizeCoefficients;

  cudaStreamCreate(&stream);
  cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
  cudaStreamAttachMemAsync(stream, modelCoefficients);
}

void CudaSegmentation::freeResources()
{
  cudaFree(index);
  // cudaFree(modelCoefficients);
  cudaStreamDestroy(stream);
}

void CudaSegmentation::realloc(unsigned int size)
{
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "REALLOC");
  cudaFree(index);
  cudaStreamSynchronize(stream);
  cudaMallocManaged(&index, sizeof(int) * size, cudaMemAttachHost);
  cudaStreamSynchronize(stream);
  cudaStreamAttachMemAsync(stream, index);
  cudaStreamSynchronize(stream);
}

// Funzione principale per segmentare i punti di input
// inputData: array nel host di float (x, y, z, intensità) × nCount
// nCount: numero di punti in input
// out_points: buffer preallocato per restituire gli inlier
// out_num_points: numero effettivo di inlier trovati
void CudaSegmentation::segment(
    float *inputData,
    int nCount,
    float **out_points,
    unsigned int *out_num_points)
{
  std::cout << "\n----------- CUDA Segmentation ---------------- "<< std::endl;
  // Inizio misurazione del tempo
  auto t1 = std::chrono::steady_clock::now();

  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Avvio segmentazione di %d punti", nCount);

  if(nCount > mall_size){
    realloc(nCount);
    mall_size = nCount;
  }

  cudaMemcpyAsync(modelCoefficients, 0, 4 * sizeof(float), cudaMemcpyHostToDevice, stream); // Inizializza i coefficienti a zero

  cudaSegmentation impl(SACMODEL_PLANE, SAC_RANSAC, stream);

  impl.set(segP);
  impl.segment(inputData, nCount, index, modelCoefficients);

  cudaDeviceSynchronize();

  // controllo coefficienti
  if (std::isnan(modelCoefficients[0]) || std::abs(modelCoefficients[3]) > 20)
  {
    std::cout << "Segmentation non valida: coefficiente[3] = " << modelCoefficients[3] << " coefficiente [2] = " << modelCoefficients[2] << " coefficiente[1] = " << modelCoefficients[1] << std::endl;
    skip = true; // Segmentation non valida, salto parte finale
  }

  if (!skip)
  {
    // 5) Raccolta degli inlier
    int idx = 0;
    // std::vector<int> inliers;
    // inliers.reserve(nCount);
    for (int i = 0; i < nCount; ++i)
    {
      if (index[i] == -1){
        // inliers.push_back(i);
        (*out_points)[4 * idx + 0] = inputData[4 * i + 0]; // x
        (*out_points)[4 * idx + 1] = inputData[4 * i + 1]; // y
        (*out_points)[4 * idx + 2] = inputData[4 * i + 2]; // z
        idx++;
      }
    }
    *out_num_points = idx;

    // Fine misurazione del tempo
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
    RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Segmentazione completata in %.3f ms", duration.count());

    // Log dei coefficienti del modello
    // RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Coefficienti modello: [%.4f, %.4f, %.4f, %.4f]",
    //             modelCoefficients[0], modelCoefficients[1], modelCoefficients[2], modelCoefficients[3]);
  }
  // Pulizia delle risorse
  // CudaSegmentation::freeResources();
  skip = false; // Reset dello stato di skip per la prossima chiamata
}