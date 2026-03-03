#include "cuda_clustering/segmentation/cuda_segmentation.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

// --------------------
// COMPACTION KERNEL
// --------------------
__global__ void compactInliersKernel(
    const float* in_points, 
    const int* index, 
    float* out_points, 
    unsigned int* d_count, 
    int max_points) 
{
    // calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // we don't want to read out of bounds
    if (tid < max_points) {
        // index != 1 means that the point is NOT part of the segmented ground plane
        if (index[tid] != 1) { 
            
            // atomically reserve a spot in the output array.
            // atomicAdd returns the old value, giving this specific thread a unique write index.
            unsigned int write_idx = atomicAdd(d_count, 1);
            
            // copy the 4 floats (X, Y, Z, Intensity) directly in VRAM
            out_points[write_idx * 4 + 0] = in_points[tid * 4 + 0];
            out_points[write_idx * 4 + 1] = in_points[tid * 4 + 1];
            out_points[write_idx * 4 + 2] = in_points[tid * 4 + 2];
            out_points[write_idx * 4 + 3] = in_points[tid * 4 + 3];
        }
    }
}


// ---------------------------------------------------------------------------------------
// RANSAC Kernels (Replacement for libcudasegmentation)
// ---------------------------------------------------------------------------------------

// Pseudo random generator
__device__ inline unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__global__ void ransacPlaneKernel(
    const float* __restrict__ points,
    int num_points,
    float threshold,
    int max_iterations,
    int* __restrict__ plane_inliers_counts,
    float4* __restrict__ plane_models,
    unsigned int seed
)
{
    // each block performs one RANSAC iteration
    int iter = blockIdx.x;
    if (iter >= max_iterations) return;

    // select 3 random points
    // use a different seed per iteration/block
    unsigned int s = seed + iter * 199999; 
    
    int idx1 = wang_hash(s) % num_points;
    int idx2 = wang_hash(s + 1) % num_points; // simple increment to get different indices
    int idx3 = wang_hash(s + 2) % num_points;

    // load points (manual indexing for float array x,y,z,i)
    float p1[3] = {points[idx1*4], points[idx1*4+1], points[idx1*4+2]};
    float p2[3] = {points[idx2*4], points[idx2*4+1], points[idx2*4+2]};
    float p3[3] = {points[idx3*4], points[idx3*4+1], points[idx3*4+2]};

    // filter out seed points that are too far away.
    // this reduces the chance of fitting to distant noise.
    if ((p1[0]*p1[0] + p1[1]*p1[1]) > 2500.0f ||
        (p2[0]*p2[0] + p2[1]*p2[1]) > 2500.0f ||
        (p3[0]*p3[0] + p3[1]*p3[1]) > 2500.0f) {
            
        if (threadIdx.x == 0) plane_inliers_counts[iter] = -1;
        return;
    }

    // compute plane model (ax + by + cz + d = 0)
    float v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    float v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};

    // cross product
    float a = v1[1] * v2[2] - v1[2] * v2[1];
    float b = v1[2] * v2[0] - v1[0] * v2[2];
    float c = v1[0] * v2[1] - v1[1] * v2[0];

    float norm = sqrtf(a*a + b*b + c*c);
    
    // check for degenerate triangle
    if (norm < 1e-6f) {
        if (threadIdx.x == 0) {
            plane_inliers_counts[iter] = -1;
        }
        return;
    }

    float inv_norm = 1.0f / norm;
    a *= inv_norm;
    b *= inv_norm;
    c *= inv_norm;
    float d = -(a * p1[0] + b * p1[1] + c * p1[2]);

    // count Inliers
    // each thread counts a subset of points
    int local_count = 0;

    for (int i = threadIdx.x; i < num_points; i += blockDim.x) {
        float x = points[i*4];
        float y = points[i*4+1];
        float z = points[i*4+2];
        
        float dist = fabsf(a * x + b * y + c * z + d);
        if (dist <= threshold) {
            local_count++;
        }
    }

    // block reduction
    __shared__ int s_counts[256]; // Block dim must be 256
    // initialize shared mem
    if (threadIdx.x < 256) s_counts[threadIdx.x] = 0;
    __syncthreads(); // Only needed if we rely on init, but we overwrite

    s_counts[threadIdx.x] = local_count;
    __syncthreads();

    // standard reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_counts[threadIdx.x] += s_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // store result
    if (threadIdx.x == 0) {
        plane_inliers_counts[iter] = s_counts[0];
        plane_models[iter] = make_float4(a, b, c, d);
    }
}

__global__ void markInliersKernel(
    const float* __restrict__ points,
    int num_points,
    int* __restrict__ indices,
    float4 best_plane,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float x = points[i*4];
    float y = points[i*4+1];
    float z = points[i*4+2];
    float dist = fabsf(best_plane.x * x + best_plane.y * y + best_plane.z * z + best_plane.w);
    
    // Mark as inlier (1) or outlier (0)
    indices[i] = (dist <= threshold) ? 1 : 0;
}

// --------------------
// CUDA SEGMENTATION
// --------------------
CudaSegmentation::CudaSegmentation(segParam_t &params)
{
  segP.distanceThreshold = params.distanceThreshold;
  segP.maxIterations = params.maxIterations;
  segP.probability = params.probability;
  segP.optimizeCoefficients = params.optimizeCoefficients;

  h_modelCoefficients.resize(4);
  d_out_count.resize(1);
}

// Funzione principale per segmentare i punti di input
// inputData: array nel host di float (x, y, z, intensità) × nCount
// nCount: numero di punti in input
// out_points: buffer preallocato per restituire gli inlier
// out_num_points: numero effettivo di inlier trovati
void CudaSegmentation::segment(
    float *inputData,
    unsigned int nCount,
    float *out_points,
    unsigned int *out_num_points,
    cudaStream_t stream)
{
  std::cout << "\n----------- CUDA Segmentation (Custom) ---------------- "
            << "\nInput point count for segmentation: " << nCount 
            << std::endl;
  
  auto t1 = std::chrono::steady_clock::now();

  if (nCount < 10) {
      *out_num_points = nCount;
      return; 
  }

  if (d_index.capacity() < nCount) {
      d_index.reserve(nCount);
      d_input.reserve(nCount * 4);
      d_output.reserve(nCount * 4);
  }
  d_index.resize(nCount);
  d_input.resize(nCount * 4);
  d_output.resize(nCount * 4);

  int* raw_index = thrust::raw_pointer_cast(d_index.data());
  float* raw_input = thrust::raw_pointer_cast(d_input.data());
  float* raw_output = thrust::raw_pointer_cast(d_output.data());

  // Copy input within GPU (inputData is already a device pointer from the controller)
  cudaMemcpyAsync(raw_input, inputData, nCount * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // ----------------------------------------------------
  // Custom RANSAC Implementation
  // ----------------------------------------------------
  std::cout << "Launching Custom RANSAC kernel on GPU..." << std::endl;

  int max_iter = segP.maxIterations;
  if (max_iter <= 0) max_iter = 100;
  if (max_iter > 1024) max_iter = 1024; // Limit for memory

  // Temp buffers for RANSAC results
  if (int(d_counts.size()) < max_iter) d_counts.resize(max_iter);
  if (int(d_planes.size()) < max_iter) d_planes.resize(max_iter);
  
  int* raw_counts = thrust::raw_pointer_cast(d_counts.data());
  float4* raw_planes = thrust::raw_pointer_cast(d_planes.data());

  // Init counts to -1
  thrust::fill(thrust::cuda::par.on(stream), d_counts.begin(), d_counts.begin() + max_iter, -1);

  // Launch RANSAC
  // Each block is 1 iteration, using 256 threads for reduction
  auto now = std::chrono::high_resolution_clock::now();
  unsigned int seed = (unsigned int)now.time_since_epoch().count();
  ransacPlaneKernel<<<max_iter, 256, 0, stream>>>(
      raw_input, nCount, (float)segP.distanceThreshold, max_iter, raw_counts, raw_planes, seed
  );
  
  // Copy RANSAC results to host to find best
  // Doing this small copy is simpler than writing a device reduction kernel for max_element with struct
  
  if (int(h_counts.size()) < max_iter) h_counts.resize(max_iter);
  if (int(h_planes.size()) < max_iter) h_planes.resize(max_iter);
  
  cudaMemcpyAsync(thrust::raw_pointer_cast(h_counts.data()), raw_counts, max_iter * sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(h_planes.data()), raw_planes, max_iter * sizeof(float4), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); // Sync to get counts

  // Find best on CPU
  int best_idx = -1;
  int max_inliers = -1;
  
  for (int i = 0; i < max_iter; i++) {
      if (h_counts[i] > max_inliers) {
          max_inliers = h_counts[i];
          best_idx = i;
      }
  }

  if (best_idx != -1 && max_inliers > 0) {
      float4 best_plane = h_planes[best_idx];
      h_modelCoefficients[0] = best_plane.x;
      h_modelCoefficients[1] = best_plane.y;
      h_modelCoefficients[2] = best_plane.z;
      h_modelCoefficients[3] = best_plane.w;
      
      // Compute points indices based on best model
      int threads = 256;
      int blocks = (nCount + threads - 1) / threads;
      markInliersKernel<<<blocks, threads, 0, stream>>>(raw_input, nCount, raw_index, best_plane, (float)segP.distanceThreshold);
      
      std::cout << "RANSAC Best: " << max_inliers << " inliers. Model: " 
                << best_plane.x << " " << best_plane.y << " " << best_plane.z << " " << best_plane.w << std::endl;
  } else {
       std::cout << "RANSAC Failed to find valid plane." << std::endl;
       skip = true;
       *out_num_points = 0;
  }

  // std::cout << "Segmentation kernel launched, retrieving results..." << std::endl;
  
  // controllo coefficienti
  if (std::isnan(h_modelCoefficients[0]) || std::abs(h_modelCoefficients[3]) > 20)
  {
    std::cout << "Segmentation failed, invalid model coefficients: [" 
              << h_modelCoefficients[0] << ", " 
              << h_modelCoefficients[1] << ", " 
              << h_modelCoefficients[2] << ", " 
              << h_modelCoefficients[3] << "]" << std::endl;
    skip = true;
    *out_num_points = 0; 
  }

  // std::cout << "Segmentation successful" << std::endl;
  
  if (!skip)
  {
    // --------------------------------------------
    // reset the GPU counter to 0
    // --------------------------------------------
    unsigned int* raw_count = thrust::raw_pointer_cast(d_out_count.data());
    cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream);

    // --------------------------------------------
    // launch the compaction kernel on the stream
    // --------------------------------------------
    int threads = 256;
    int blocks = (nCount + threads - 1) / threads;
    // USE DEVICE POINTERS HERE: raw_input for read, raw_output for write
    compactInliersKernel<<<blocks, threads, 0, stream>>>(raw_input, raw_index, raw_output, raw_count, nCount);

    // --------------------------------------------
    // copy the final count back to the CPU
    // --------------------------------------------
    cudaMemcpyAsync(out_num_points, raw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        
    // ------------------------------------------------------------------------------
    // wait for the compaction and copy to finish before returning to the main node
    // ------------------------------------------------------------------------------
    cudaStreamSynchronize(stream); 

    // Copy result within GPU (out_points is a device pointer from the controller)
    if (*out_num_points > 0) {
        cudaMemcpyAsync(out_points, raw_output, (*out_num_points) * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    std::cout << "Segmentation completed in " << duration / 1e6 << " ms, found " << *out_num_points << " inliers"
              << "\n------------------------------------------------------- \n" << std::endl;
  }

  // Pulizia delle risorse
  // CudaSegmentation::freeResources();
  skip = false; // Reset dello stato di skip per la prossima chiamata
  // RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Returning from segment");
}