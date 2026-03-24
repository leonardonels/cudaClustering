#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdint>
#include <algorithm>
#include <cstdio>

#include "cuda_clustering/segmentation/gpu_kernels_fixed.hpp"

// ==============================================================================
// POINT STRUCTURE
// ==============================================================================
struct Point
{
    float x, y, z;
};

namespace {

bool selectCudaDeviceOrFallback(int preferred_device, int* selected_device) {
    int device_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        std::fprintf(
            stderr,
            "[CudaSegmentationKernel] cudaGetDeviceCount failed or no devices available: %s\n",
            cudaGetErrorString(count_err));
        return false;
    }

    int target_device = preferred_device;
    if (target_device < 0 || target_device >= device_count) {
        target_device = 0;
    }

    int current_device = -1;
    cudaError_t get_err = cudaGetDevice(&current_device);
    bool need_set_device =
        (get_err != cudaSuccess) ||
        (current_device < 0) ||
        (current_device >= device_count) ||
        (current_device != target_device);

    if (need_set_device) {
        // Clear pending runtime error state before trying to recover.
        cudaGetLastError();

        cudaError_t set_err = cudaSetDevice(target_device);
        if (set_err != cudaSuccess) {
            if (target_device != 0) {
                set_err = cudaSetDevice(0);
                if (set_err == cudaSuccess) {
                    target_device = 0;
                }
            }

            if (set_err != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "[CudaSegmentationKernel] Failed to set CUDA device (preferred=%d): %s\n",
                    preferred_device,
                    cudaGetErrorString(set_err));
                return false;
            }
        }
    }

    if (selected_device) {
        *selected_device = target_device;
    }

    return true;
}

struct FixedSegScratch {
    thrust::device_vector<Point> d_points;
    thrust::device_vector<uint8_t> d_is_ground;
    thrust::device_vector<int> d_indices;
    float* d_sums = nullptr;
    float4* d_plane = nullptr;
    unsigned int* d_out_count = nullptr;
};

thread_local FixedSegScratch g_fixed_seg_scratch;

bool ensureScratchBuffers() {
    if (g_fixed_seg_scratch.d_sums == nullptr) {
        cudaError_t err = cudaMalloc(&g_fixed_seg_scratch.d_sums, 10 * sizeof(float));
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[CudaSegmentationKernel] cudaMalloc d_sums failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    if (g_fixed_seg_scratch.d_plane == nullptr) {
        cudaError_t err = cudaMalloc(&g_fixed_seg_scratch.d_plane, sizeof(float4));
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[CudaSegmentationKernel] cudaMalloc d_plane failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    if (g_fixed_seg_scratch.d_out_count == nullptr) {
        cudaError_t err = cudaMalloc(&g_fixed_seg_scratch.d_out_count, sizeof(unsigned int));
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[CudaSegmentationKernel] cudaMalloc d_out_count failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    return true;
}

} // namespace

// ==============================================================================
// KERNEL 1: Calculate Covariance Matrix Statistics
// ==============================================================================
__global__ void computeCovarianceStats(
    const Point* __restrict__ points,
    const uint8_t* __restrict__ is_ground,
    int num_points,
    bool use_mask,
    int limit_count,
    float* __restrict__ global_sums 
) {
    // 1. Setup Shared Memory
    __shared__ float s_sums[10];
    
    int tid = threadIdx.x;
    if (tid < 10) s_sums[tid] = 0.0f;
    __syncthreads();

    // 2. Main Loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        bool include_point = false;
        if (!use_mask) {
            if (idx < limit_count) include_point = true;
        } else {
            if (is_ground[idx]) include_point = true;
        }

        if (include_point) {
            Point p = points[idx];
            atomicAdd(&s_sums[0], p.x);
            atomicAdd(&s_sums[1], p.y);
            atomicAdd(&s_sums[2], p.z);
            atomicAdd(&s_sums[3], p.x * p.x);
            atomicAdd(&s_sums[4], p.x * p.y);
            atomicAdd(&s_sums[5], p.x * p.z);
            atomicAdd(&s_sums[6], p.y * p.y);
            atomicAdd(&s_sums[7], p.y * p.z);
            atomicAdd(&s_sums[8], p.z * p.z);
            atomicAdd(&s_sums[9], 1.0f);
        }
    }
    __syncthreads();

    // 3. Write to Global Memory
    if (tid < 10) {
        atomicAdd(&global_sums[tid], s_sums[tid]);
    }
}

// ==============================================================================
// KERNEL 2: Classify Points based on Plane Equation
// ==============================================================================
__global__ void classifyPointsKernel(
    const Point* __restrict__ points,
    uint8_t* __restrict__ is_ground,
    int num_points,
    const float4* __restrict__ plane_ptr, 
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Read the plane from GPU memory
    float4 plane = *plane_ptr; 

    Point p = points[idx];
    float dist = fabsf(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w);
    
    is_ground[idx] = (dist < threshold) ? 1u : 0u;
}

__global__ void packPointsKernel(
    const float* __restrict__ input_data,
    Point* __restrict__ points,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    points[idx].x = input_data[4 * idx + 0];
    points[idx].y = input_data[4 * idx + 1];
    points[idx].z = input_data[4 * idx + 2];
}

__global__ void collectNonGroundKernel(
    const float* __restrict__ input_data,
    const uint8_t* __restrict__ is_ground,
    const int* __restrict__ sorted_to_original,
    int num_points,
    float* __restrict__ out_points,
    unsigned int* __restrict__ out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points || is_ground[idx] != 0u) {
        return;
    }

    int original_idx = sorted_to_original[idx];
    unsigned int out_idx = atomicAdd(out_count, 1u);
    out_points[4 * out_idx + 0] = input_data[4 * original_idx + 0];
    out_points[4 * out_idx + 1] = input_data[4 * original_idx + 1];
    out_points[4 * out_idx + 2] = input_data[4 * original_idx + 2];
    out_points[4 * out_idx + 3] = input_data[4 * original_idx + 3];
}

// ==============================================================================
// HELPER FUNCTION: 3x3 Determinant
// ==============================================================================
__device__ inline float det3x3(float xx, float xy, float xz, 
                               float yy, float yz, float zz) {
    return xx * (yy * zz - yz * yz) - 
           xy * (xy * zz - yz * xz) + 
           xz * (xy * yz - yy * xz);
}

// ==============================================================================
// KERNEL 3: Solve Plane using Eigenvalue Decomposition
// ==============================================================================
__global__
void solvePlaneDevice(float* sums, float4* plane) {
    if (threadIdx.x != 0) return;
    
    float N = sums[9];
    if (N < 3) {
        // Not enough points to form a plane
        *plane = make_float4(0, 0, 1, 0); 
        return;
    }

    // 1. Compute Centroid
    float3 centroid = {sums[0]/N, sums[1]/N, sums[2]/N};

    // 2. Construct Covariance Matrix (Upper Triangle of Symmetric Matrix)
    // Var(X) = E[X^2] - E[X]^2
    float xx = sums[3]/N - centroid.x * centroid.x;
    float xy = sums[4]/N - centroid.x * centroid.y;
    float xz = sums[5]/N - centroid.x * centroid.z;
    float yy = sums[6]/N - centroid.y * centroid.y;
    float yz = sums[7]/N - centroid.y * centroid.z;
    float zz = sums[8]/N - centroid.z * centroid.z;

    // -----------------------------------------------------------
    // 3. ANALYTIC EIGENVALUE SOLVER (Robust 3x3)
    // -----------------------------------------------------------
    
    // Scale the matrix by its max element to avoid numerical issues with small/large coordinates
    float scale = fmaxf(fabsf(xx), fmaxf(fabsf(yy), fabsf(zz)));
    if (scale < 1e-6f) {
        // Variance is zero (points are all in one spot)
        *plane = make_float4(0, 0, 1, -centroid.z);
        return;
    }

    // Normalize matrix
    xx /= scale; xy /= scale; xz /= scale;
    yy /= scale; yz /= scale; zz /= scale;

    // Characteristic Equation: lambda^3 + a*lambda^2 + b*lambda + c = 0
    // For a symmetric matrix M:
    // a = -trace(M)
    // b = 0.5 * (trace(M)^2 - trace(M^2))
    // c = -det(M)
    
    float trace_A = xx + yy + zz;
    float trace_A2 = (xx*xx + xy*xy + xz*xz) + 
                     (xy*xy + yy*yy + yz*yz) + 
                     (xz*xz + yz*yz + zz*zz);
    
    float a = -trace_A;
    float b = 0.5f * (trace_A * trace_A - trace_A2);
    float c = -det3x3(xx, xy, xz, yy, yz, zz);

    // Solve cubic equation using trigonometric method
    float p = b - a * a / 3.0f;
    float q = 2.0f * a * a * a / 27.0f - a * b / 3.0f + c;
    float p3 = p * p * p;
    float D = 4.0f * p3 + 27.0f * q * q;

    // Smallest eigenvalue (min_lambda)
    float min_lambda;
    if (D >= 0) {
        // This case is rare for covariance matrices (usually 3 real roots),
        // but can happen if eigenvalues are identical. 
        // Fallback to a safe small value or 0.
        min_lambda = 0.0f; 
    } else {
        float r = sqrtf(-4.0f * p / 3.0f);
        float phi = acosf(-4.0f * q / (r * r * r)) / 3.0f;
        // The roots are sorted or periodic. For covariance matrices,
        // the smallest root is typically found at this offset:
        min_lambda = r * cosf(phi + 2.0f * 3.14159265f / 3.0f) - a / 3.0f;
    }

    // -----------------------------------------------------------
    // 4. COMPUTE EIGENVECTOR (Normal Vector)
    // Solve (M - min_lambda * I) * v = 0
    // The eigenvector is the cross product of any two independent rows of (M - lambda*I)
    // -----------------------------------------------------------
    
    float l = min_lambda;
    float3 r0 = {xx - l, xy,     xz};
    float3 r1 = {xy,     yy - l, yz};
    float3 r2 = {xz,     yz,     zz - l};

    // Calculate cross products of rows
    float3 v0 = {r0.y * r1.z - r0.z * r1.y,  r0.z * r1.x - r0.x * r1.z,  r0.x * r1.y - r0.y * r1.x};
    float3 v1 = {r0.y * r2.z - r0.z * r2.y,  r0.z * r2.x - r0.x * r2.z,  r0.x * r2.y - r0.y * r2.x};
    float3 v2 = {r1.y * r2.z - r1.z * r2.y,  r1.z * r2.x - r1.x * r2.z,  r1.x * r2.y - r1.y * r2.x};

    // Pick the most robust (longest) vector to avoid precision loss if rows are nearly parallel
    float d0 = v0.x*v0.x + v0.y*v0.y + v0.z*v0.z;
    float d1 = v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
    float d2 = v2.x*v2.x + v2.y*v2.y + v2.z*v2.z;

    float3 normal;
    float norm_sq;
    if (d0 >= d1 && d0 >= d2) { normal = v0; norm_sq = d0; }
    else if (d1 >= d2)        { normal = v1; norm_sq = d1; }
    else                      { normal = v2; norm_sq = d2; }

    // Normalize
    if (norm_sq < 1e-6f) {
        normal = make_float3(0, 0, 1); // Fallback
    } else {
        float inv_norm = rsqrtf(norm_sq);
        normal.x *= inv_norm;
        normal.y *= inv_norm;
        normal.z *= inv_norm;
    }

    // Ensure normal points "Up" (positive Z)
    // This is standard for ground plane segmentation
    if (normal.z < 0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }

    // 5. Compute 'd' in ax + by + cz + d = 0
    // d = -dot(normal, centroid)
    float d_val = -(normal.x * centroid.x + normal.y * centroid.y + normal.z * centroid.z);

    *plane = make_float4(normal.x, normal.y, normal.z, d_val);
}

// ==============================================================================
// BATCH PROCESSING FUNCTION (HOST)
// ==============================================================================
/**
 * @brief Process a batch of points through complete segmentation pipeline
 * 
 * @param d_points Device vector of points to segment
 * @param d_is_ground [out] Device vector marking ground/non-ground classification
 * @param d_sums Temporary buffer for covariance statistics
 * @param num_iter Number of refinement iterations
 * @param num_lpr Initial number of seed/lower points to use
 * @param th_dist Distance threshold for plane classification
 * @param stream CUDA stream for async execution
 */
void processBatch(
    thrust::device_vector<Point>& d_points,
    thrust::device_vector<uint8_t>& d_is_ground,
    thrust::device_vector<int>& d_indices,
    float4* d_plane,
    float* d_sums,
    int num_iter,
    int num_lpr,
    float th_dist_fit,
    float th_dist_final,
    cudaStream_t stream
) {
    int num_points = d_points.size();
    if (num_points == 0) return;

    // Use Thrust with Execution Policy to bind to the stream
    auto policy = thrust::cuda::par.on(stream);

    // 1. Sort points by Z coordinate while preserving original indices.
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_points.begin(), d_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_points.end(), d_indices.end()));
    thrust::sort(policy, zip_begin, zip_end,
                 [] __host__ __device__ (const thrust::tuple<Point, int>& a,
                                         const thrust::tuple<Point, int>& b) {
                     return thrust::get<0>(a).z < thrust::get<0>(b).z;
                 });

    // Define grid/block dimensions
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    // 2. Iterative refinement loop
    for (int i = 0; i < num_iter; i++) {
        // Reset sums (Async)
        cudaMemsetAsync(d_sums, 0, 10 * sizeof(float), stream);

        // Kernel 1: Compute covariance statistics
        // use_mask = true for iterations > 0 (refine with previously classified points)
        computeCovarianceStats<<<blocks, threads, 10*sizeof(float), stream>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            (i > 0),      // use_mask: false for first iteration, true for refinement
            num_lpr,      // limit_count: only use first num_lpr points in initial iteration
            d_sums
        );

        // Kernel 2: Solve plane using eigenvalue decomposition
        solvePlaneDevice<<<1, 1, 0, stream>>>(d_sums, d_plane);

        // Kernel 3: Classify points as ground/non-ground
        classifyPointsKernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            d_plane,
            th_dist_fit
        );
    }

    // Final relaxed pass: include rough terrain/grass around the fitted plane
    classifyPointsKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_is_ground.data()),
        num_points,
        d_plane,
        th_dist_final
    );

}

void segmentPointsFixedCuda(
    float* input_data,
    int num_points,
    float* out_points,
    unsigned int* out_num_points,
    int num_iter,
    int num_lpr,
    float th_dist_fit,
    float th_dist_final,
    cudaStream_t stream
) {
    if (num_points <= 0) {
        *out_num_points = 0;
        return;
    }

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    if (!ensureScratchBuffers()) {
        *out_num_points = 0;
        return;
    }

    g_fixed_seg_scratch.d_points.resize(num_points);
    g_fixed_seg_scratch.d_is_ground.resize(num_points);
    g_fixed_seg_scratch.d_indices.resize(num_points);

    thrust::sequence(
        thrust::cuda::par.on(stream),
        g_fixed_seg_scratch.d_indices.begin(),
        g_fixed_seg_scratch.d_indices.end());

    packPointsKernel<<<blocks, threads, 0, stream>>>(
        input_data,
        thrust::raw_pointer_cast(g_fixed_seg_scratch.d_points.data()),
        num_points
    );

    processBatch(
        g_fixed_seg_scratch.d_points,
        g_fixed_seg_scratch.d_is_ground,
        g_fixed_seg_scratch.d_indices,
        g_fixed_seg_scratch.d_plane,
        g_fixed_seg_scratch.d_sums,
        num_iter,
        num_lpr,
        th_dist_fit,
        th_dist_final,
        stream
    );

    cudaMemsetAsync(g_fixed_seg_scratch.d_out_count, 0, sizeof(unsigned int), stream);

    collectNonGroundKernel<<<blocks, threads, 0, stream>>>(
        input_data,
        thrust::raw_pointer_cast(g_fixed_seg_scratch.d_is_ground.data()),
        thrust::raw_pointer_cast(g_fixed_seg_scratch.d_indices.data()),
        num_points,
        out_points,
        g_fixed_seg_scratch.d_out_count
    );

    unsigned int h_out_count = 0;
    cudaMemcpyAsync(&h_out_count, g_fixed_seg_scratch.d_out_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    *out_num_points = h_out_count;
}

// ==============================================================================
// NEW CLASS IMPLEMENTATION
// ==============================================================================

CudaSegmentationKernel::CudaSegmentationKernel(segParam_t &params)
{
  segP.distanceThreshold = params.distanceThreshold;
  segP.maxIterations = params.maxIterations;
  segP.probability = params.probability;
  segP.optimizeCoefficients = params.optimizeCoefficients;

    if (!selectCudaDeviceOrFallback(-1, &device_id)) {
        stream = nullptr;
        return;
    }

    cudaError_t stream_err = cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess) {
        std::fprintf(
                stderr,
                "[CudaSegmentationKernel] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(stream_err));
        stream = nullptr;
    }
}

CudaSegmentationKernel::~CudaSegmentationKernel() {
        if (stream != nullptr) {
                cudaStreamDestroy(stream);
        }
}

void CudaSegmentationKernel::segment(float *inputData,
                 int nCount,
                 float **out_points,
                 unsigned int *out_num_points
) {
    if (!out_points || !*out_points || !out_num_points) {
        return;
    }

    int previous_device = device_id;
    int selected_device = device_id;
    if (!selectCudaDeviceOrFallback(device_id, &selected_device)) {
        *out_num_points = 0;
        return;
    }

    if (stream != nullptr && selected_device != previous_device) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    device_id = selected_device;

    if (stream == nullptr) {
        cudaError_t stream_err = cudaStreamCreate(&stream);
        if (stream_err != cudaSuccess) {
            std::fprintf(
                stderr,
                "[CudaSegmentationKernel] cudaStreamCreate failed in segment(): %s\n",
                cudaGetErrorString(stream_err));
            *out_num_points = 0;
            return;
        }
    }

    constexpr int kFixedKernelMaxIter = 16;
    int requested_iter = std::max(1, segP.maxIterations);
    int num_iter = std::min(requested_iter, kFixedKernelMaxIter);
    if (requested_iter > kFixedKernelMaxIter) {
        static bool warned_once = false;
        if (!warned_once) {
            std::fprintf(
                stderr,
                "[CudaSegmentationKernel] Capping maxIterations from %d to %d for fixed CUDA path performance.\n",
                requested_iter,
                kFixedKernelMaxIter);
            warned_once = true;
        }
    }

    int num_lpr = std::max(1, std::min(nCount, 200));

    float fit_threshold = static_cast<float>(segP.distanceThreshold);
    // Keep fitting strict and only relax final labeling to better absorb grass.
    float final_threshold = std::max(fit_threshold, fit_threshold * 1.5f);

    segmentPointsFixedCuda(
        inputData,
        nCount,
        *out_points,
        out_num_points,
        num_iter,
        num_lpr,
        fit_threshold,
        final_threshold,
        stream);
}