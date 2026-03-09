#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/utils/cached_allocator.hpp"

#include <iostream>
#include <algorithm>
#include <vector>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ==========================================================================
//  KERNEL 1:  Bounding box reduction (GPU — no D→H copy needed)
// ==========================================================================
//  each block reduces its chunk to local min/max, then atomicMin/Max on
//  global output.  We use __int_as_float / __float_as_int trick for
//  atomics on floats.
// ==========================================================================
__device__ inline void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ inline void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// d_bbox layout: [minX, minY, minZ, maxX, maxY, maxZ]
__global__ void boundingBoxKernel(
    const float* __restrict__ points,
    unsigned int nPoints,
    float* __restrict__ d_bbox)
{
    __shared__ float s_min[3][256];  // blockDim.x capped to 256 for shared mem
    __shared__ float s_max[3][256];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float lminX = 1e30f, lminY = 1e30f, lminZ = 1e30f;
    float lmaxX = -1e30f, lmaxY = -1e30f, lmaxZ = -1e30f;

    // grid-stride loop
    for (unsigned int i = gid; i < nPoints; i += blockDim.x * gridDim.x) {
        float x = points[i * 4 + 0];
        float y = points[i * 4 + 1];
        float z = points[i * 4 + 2];
        lminX = fminf(lminX, x); lmaxX = fmaxf(lmaxX, x);
        lminY = fminf(lminY, y); lmaxY = fmaxf(lmaxY, y);
        lminZ = fminf(lminZ, z); lmaxZ = fmaxf(lmaxZ, z);
    }

    s_min[0][tid] = lminX; s_min[1][tid] = lminY; s_min[2][tid] = lminZ;
    s_max[0][tid] = lmaxX; s_max[1][tid] = lmaxY; s_max[2][tid] = lmaxZ;
    __syncthreads();

    // shared memory reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[0][tid] = fminf(s_min[0][tid], s_min[0][tid + stride]);
            s_min[1][tid] = fminf(s_min[1][tid], s_min[1][tid + stride]);
            s_min[2][tid] = fminf(s_min[2][tid], s_min[2][tid + stride]);
            s_max[0][tid] = fmaxf(s_max[0][tid], s_max[0][tid + stride]);
            s_max[1][tid] = fmaxf(s_max[1][tid], s_max[1][tid + stride]);
            s_max[2][tid] = fmaxf(s_max[2][tid], s_max[2][tid + stride]);
        }
        __syncthreads();
    }

    // block winner atomically updates global bbox
    if (tid == 0) {
        atomicMinFloat(&d_bbox[0], s_min[0][0]);
        atomicMinFloat(&d_bbox[1], s_min[1][0]);
        atomicMinFloat(&d_bbox[2], s_min[2][0]);
        atomicMaxFloat(&d_bbox[3], s_max[0][0]);
        atomicMaxFloat(&d_bbox[4], s_max[1][0]);
        atomicMaxFloat(&d_bbox[5], s_max[2][0]);
    }
}

// ==========================================================================
//  KERNEL 2:  Compute a voxel hash for every point
// ==========================================================================
//  reads bounding box from device pointer (output of boundingBoxKernel)
//  Hash = ix + iy * GRID + iz * GRID * GRID
// ==========================================================================
__global__ void computeVoxelKeysKernel(
    const float* __restrict__ points,
    int* __restrict__ keys,
    unsigned int nPoints,
    float voxelX, float voxelY, float voxelZ,
    const float* __restrict__ d_bbox,  // [minX, minY, minZ, maxX, maxY, maxZ]
    int* __restrict__ d_grid)          // output: [gridX, gridY] computed by thread 0
{
    // thread 0 computes grid dims from device-side bbox and writes to d_grid
    __shared__ float s_bbox[6];
    __shared__ int s_grid[2];

    if (threadIdx.x < 6) s_bbox[threadIdx.x] = d_bbox[threadIdx.x];
    __syncthreads();

    float minX = s_bbox[0], minY = s_bbox[1], minZ = s_bbox[2];

    if (threadIdx.x == 0) {
        float maxX = s_bbox[3], maxY = s_bbox[4];
        s_grid[0] = __float2int_ru((maxX - minX) / voxelX) + 1;
        s_grid[1] = __float2int_ru((maxY - minY) / voxelY) + 1;
        // also store to global so subsequent kernels can read gridX/gridY
        d_grid[0] = s_grid[0];
        d_grid[1] = s_grid[1];
    }
    __syncthreads();

    int gridX = s_grid[0];
    int gridY = s_grid[1];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    float x = points[tid * 4 + 0];
    float y = points[tid * 4 + 1];
    float z = points[tid * 4 + 2];

    int ix = __float2int_rd((x - minX) / voxelX);
    int iy = __float2int_rd((y - minY) / voxelY);
    int iz = __float2int_rd((z - minZ) / voxelZ);

    keys[tid] = ix + iy * gridX + iz * gridX * gridY;
}

// ==========================================================================
//  Helper:  union-find with path compression (device)
// ==========================================================================
__device__ int uf_find(int* parent, int i)
{
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];   // path splitting
        i = parent[i];
    }
    return i;
}

__device__ void uf_union(int* parent, int a, int b)
{
    while (true) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b) return;
        // Smaller root becomes child — deterministic to avoid races
        if (a > b) { int tmp = a; a = b; b = tmp; }
        int old = atomicCAS(&parent[b], b, a);
        if (old == b) return;   // success
        // Retry with updated roots
    }
}

// ==========================================================================
//  KERNEL 3:  Union-find on voxel grid — 26-connectivity
// ==========================================================================
//  reads gridX/gridY from device pointer (computed by computeVoxelKeysKernel)
// ==========================================================================
__global__ void unionFindKernel(
    const int* __restrict__ uniqueKeys,   // sorted unique voxel hashes
    int  numVoxels,
    int* __restrict__ parent,
    const int* __restrict__ d_grid)       // [gridX, gridY]
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)numVoxels) return;

    int gridX = d_grid[0];
    int gridY = d_grid[1];

    int myKey = uniqueKeys[tid];
    int iz = myKey / (gridX * gridY);
    int rem = myKey % (gridX * gridY);
    int iy = rem / gridX;
    int ix = rem % gridX;

    // 26-connected neighbours
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;
                if (nx < 0 || ny < 0 || nz < 0) continue;

                int nKey = nx + ny * gridX + nz * gridX * gridY;

                // Binary-search for nKey in uniqueKeys
                int lo = 0, hi = numVoxels - 1;
                int found = -1;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    int mk  = uniqueKeys[mid];
                    if (mk == nKey) { found = mid; break; }
                    else if (mk < nKey) lo = mid + 1;
                    else                hi = mid - 1;
                }
                if (found >= 0) {
                    uf_union(parent, (int)tid, found);
                }
            }
        }
    }
}

// ==========================================================================
//  KERNEL 4:  Flatten parent array (path compress to root)
// ==========================================================================
__global__ void flattenParentKernel(int* parent, int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)n) return;
    parent[tid] = uf_find(parent, tid);
}

// ==========================================================================
//  KERNEL 5:  Assign cluster label to every point via sorted-key lookup
// ==========================================================================
//  each point already has a voxel key in d_voxelKeys.
//  we binary-search in the *filtered* unique keys to find its voxel index,
//  then read the flattened parent (= cluster root label).
//  Points whose voxel was filtered out get label -1.
// ==========================================================================
__global__ void assignClusterLabelsKernel(
    const int* __restrict__ pointKeys,      // voxel key per original point
    unsigned int nPoints,
    const int* __restrict__ filteredKeys,    // sorted filtered unique voxel hashes
    int numFiltered,
    const int* __restrict__ parent,          // flattened parent array (indexed by filtered-voxel-idx)
    int* __restrict__ pointLabels)           // output: cluster root label per point (-1 if filtered)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    int key = pointKeys[tid];

    // binary search in filteredKeys
    int lo = 0, hi = numFiltered - 1;
    int found = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int mk = filteredKeys[mid];
        if (mk == key) { found = mid; break; }
        else if (mk < key) lo = mid + 1;
        else               hi = mid - 1;
    }

    pointLabels[tid] = (found >= 0) ? parent[found] : -1;
}

// ==========================================================================
//  KERNEL 6:  Per-cluster bounding box via atomics
// ==========================================================================
//  each point contributes to its cluster's bounding box.
//  d_clusterBBox layout per cluster: [minX, minY, minZ, maxX, maxY, maxZ]
//  d_clusterSizes: atomically counted per cluster
// ==========================================================================
__global__ void clusterBBoxKernel(
    const float* __restrict__ points,
    const int* __restrict__ pointLabels,    // cluster root label per point
    const int* __restrict__ labelMap,        // root_label → compact cluster id
    int numLabels,                           // number of unique labels (for binary search in labelMap)
    const int* __restrict__ labelKeys,       // sorted unique root labels
    unsigned int nPoints,
    float* __restrict__ d_clusterBBox,       // [6 * numClusters]
    unsigned int* __restrict__ d_clusterSizes)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    int label = pointLabels[tid];
    if (label < 0) return;

    // binary search for label in labelKeys to get compact cluster id
    int lo = 0, hi = numLabels - 1;
    int cid = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int mk = labelKeys[mid];
        if (mk == label) { cid = labelMap[mid]; break; }
        else if (mk < label) lo = mid + 1;
        else                  hi = mid - 1;
    }
    if (cid < 0) return;

    float x = points[tid * 4 + 0];
    float y = points[tid * 4 + 1];
    float z = points[tid * 4 + 2];

    atomicMinFloat(&d_clusterBBox[cid * 6 + 0], x);
    atomicMinFloat(&d_clusterBBox[cid * 6 + 1], y);
    atomicMinFloat(&d_clusterBBox[cid * 6 + 2], z);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 3], x);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 4], y);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 5], z);

    atomicAdd(&d_clusterSizes[cid], 1u);
}

// ==========================================================================
//  KERNEL 7:  Dimension filter — check each cluster's bbox on GPU
// ==========================================================================
//  produces a compacted list of cone center points (x,y,z) for valid clusters.
// ==========================================================================
__global__ void dimensionFilterKernel(
    const float* __restrict__ d_clusterBBox,   // [6 * numClusters]
    const unsigned int* __restrict__ d_clusterSizes,
    int numClusters,
    unsigned int minClusterSize,
    unsigned int maxClusterSize,
    float filterMinX, float filterMinY, float filterMinZ,
    float filterMaxX, float filterMaxY, float filterMaxZ,
    float maxHeight,
    float* __restrict__ d_conePoints,          // output: [3 * numClusters] max
    unsigned int* __restrict__ d_numCones)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)numClusters) return;

    unsigned int sz = d_clusterSizes[tid];
    if (sz < minClusterSize || sz > maxClusterSize) return;

    float mnX = d_clusterBBox[tid * 6 + 0];
    float mnY = d_clusterBBox[tid * 6 + 1];
    float mnZ = d_clusterBBox[tid * 6 + 2];
    float mxX = d_clusterBBox[tid * 6 + 3];
    float mxY = d_clusterBBox[tid * 6 + 4];
    float mxZ = d_clusterBBox[tid * 6 + 5];

    float dx = mxX - mnX;
    float dy = mxY - mnY;
    float dz = mxZ - mnZ;

    // isCone check
    if (mnZ < maxHeight &&
        dx < filterMaxX && dy < filterMaxY && dz < filterMaxZ &&
        dx > filterMinX && dy > filterMinY && dz > filterMinZ)
    {
        unsigned int idx = atomicAdd(d_numCones, 1u);
        d_conePoints[idx * 3 + 0] = (mxX + mnX) * 0.5f;
        d_conePoints[idx * 3 + 1] = (mxY + mnY) * 0.5f;
        d_conePoints[idx * 3 + 2] = (mxZ + mnZ) * 0.5f;
    }
}

// ==========================================================================
//  CudaClustering implementation
// ==========================================================================
CudaClustering::CudaClustering(clustering_parameters& param)
{
    ecp.minClusterSize  = param.clustering.minClusterSize;
    ecp.maxClusterSize  = param.clustering.maxClusterSize;
    ecp.voxelX          = param.clustering.voxelX;
    ecp.voxelY          = param.clustering.voxelY;
    ecp.voxelZ          = param.clustering.voxelZ;
    ecp.countThreshold  = param.clustering.countThreshold;

    filterParams = param.filtering;
    cudaStreamCreate(&stream);

    // pre-allocate small fixed-size device buffers
    d_bbox.resize(6);       // [minX, minY, minZ, maxX, maxY, maxZ]
    d_grid.resize(2);       // [gridX, gridY]
    d_numCones.resize(1);
}

CudaClustering::~CudaClustering()
{
    if (stream != NULL) cudaStreamDestroy(stream);
}

void CudaClustering::getInfo()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "----device id: %d info----", i);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  GPU : %s", prop.name);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  Capability: %d.%d", prop.major, prop.minor);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  Global memory: %luMB", prop.totalGlobalMem >> 20);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  SM in a block: %luKB", prop.sharedMemPerBlock >> 10);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  warp size: %d", prop.warpSize);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  threads in a block: %d", prop.maxThreadsPerBlock);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  block dim: (%d,%d,%d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "  grid dim: (%d,%d,%d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

// --------------------------------------------------------------------------
//  extractClusters — full GPU pipeline (single D→H sync at the end)
// --------------------------------------------------------------------------
//  1. Bounding box                              (GPU kernel)
//  2. Compute voxel hash per point              (GPU kernel, reads bbox from device)
//  3. Sort points by voxel hash                 (GPU — thrust)
//  4. Reduce to unique voxels + per-voxel counts(GPU — thrust)
//  5. Filter voxels by countThreshold           (GPU — thrust::copy_if)
//  6. Union-find 26-connectivity                (GPU kernel)
//  7. Flatten parent labels                     (GPU kernel)
//  8. Assign cluster label per point            (GPU kernel)
//  9. Per-cluster bounding box via atomics       (GPU kernel)
//  10. Dimension filter → cone points           (GPU kernel)
//  11. Copy cone points D→H (tiny)             (single sync)
// --------------------------------------------------------------------------
void CudaClustering::extractClusters(
    float* input,             // device pointer  (x,y,z,i)*N
    unsigned int inputSize,
    float* /*outputEC*/,      // device pointer  (unused in this pipeline)
    std::shared_ptr<visualization_msgs::msg::Marker> cones)
{
    std::cout << "\n------------ CUDA Clustering (Custom) ----------------" << std::endl;
    auto t1 = std::chrono::steady_clock::now();

    if (inputSize < ecp.minClusterSize) {
        RCLCPP_WARN(rclcpp::get_logger("clustering_node"),
                     "Not enough points for clustering (%u < %u)", inputSize, ecp.minClusterSize);
        return;
    }

    const int threads = 1024;
    const int bbThreads = 256;  // bounded for shared memory in bbox kernel
    int blocks;

    // ------------------------------------------------------------------
    // 1. Bounding box on GPU (no D→H copy)
    // ------------------------------------------------------------------
    // init bbox: min channels to +inf, max channels to -inf
    float bbox_init[6] = {1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_bbox.data()), bbox_init,
                    6 * sizeof(float), cudaMemcpyHostToDevice, stream);

    blocks = std::min((int)((inputSize + bbThreads - 1) / bbThreads), 256);
    boundingBoxKernel<<<blocks, bbThreads, 0, stream>>>(
        input, inputSize, thrust::raw_pointer_cast(d_bbox.data()));

    // ------------------------------------------------------------------
    // 2. Compute voxel hash per point (reads bbox + writes gridX/gridY from/to device)
    // ------------------------------------------------------------------
    if (d_voxelKeys.capacity() < inputSize) {
        d_voxelKeys.reserve(inputSize);
        d_sortedIndices.reserve(inputSize);
    }
    d_voxelKeys.resize(inputSize);
    d_sortedIndices.resize(inputSize);

    blocks = (inputSize + threads - 1) / threads;
    computeVoxelKeysKernel<<<blocks, threads, 0, stream>>>(
        input, thrust::raw_pointer_cast(d_voxelKeys.data()), inputSize,
        ecp.voxelX, ecp.voxelY, ecp.voxelZ,
        thrust::raw_pointer_cast(d_bbox.data()),
        thrust::raw_pointer_cast(d_grid.data()));

    // ------------------------------------------------------------------
    // 3. Sort by voxel key (GPU) — sort in-place, no extra copy
    // ------------------------------------------------------------------
    thrust::sequence(thrust::cuda::par(alloc).on(stream),
                     d_sortedIndices.begin(), d_sortedIndices.end());
    thrust::sort_by_key(thrust::cuda::par(alloc).on(stream),
                        d_voxelKeys.begin(), d_voxelKeys.end(),
                        d_sortedIndices.begin());

    // ------------------------------------------------------------------
    // 4. Reduce to unique voxels + counts (GPU)
    // ------------------------------------------------------------------
    if (d_uniqueKeys.capacity() < inputSize)  d_uniqueKeys.reserve(inputSize);
    if (d_voxelCounts.capacity() < inputSize) d_voxelCounts.reserve(inputSize);
    d_uniqueKeys.resize(inputSize);
    d_voxelCounts.resize(inputSize);

    auto new_end = thrust::reduce_by_key(
        thrust::cuda::par(alloc).on(stream),
        d_voxelKeys.begin(), d_voxelKeys.end(),
        thrust::make_constant_iterator(1u),
        d_uniqueKeys.begin(),
        d_voxelCounts.begin());

    int numVoxels = (int)(new_end.first - d_uniqueKeys.begin());
    d_uniqueKeys.resize(numVoxels);
    d_voxelCounts.resize(numVoxels);

    // ------------------------------------------------------------------
    // 5. Filter voxels by countThreshold (GPU — thrust::copy_if)
    // ------------------------------------------------------------------
    if ((int)d_filteredKeys.capacity() < numVoxels)
        d_filteredKeys.reserve(numVoxels);
    d_filteredKeys.resize(numVoxels);

    int countThresh = ecp.countThreshold;
    auto filt_end = thrust::copy_if(
        thrust::cuda::par(alloc).on(stream),
        d_uniqueKeys.begin(), d_uniqueKeys.end(),
        d_voxelCounts.begin(),  // stencil
        d_filteredKeys.begin(),
        [countThresh] __device__ (unsigned int c) { return (int)c >= countThresh; });

    int numFiltered = (int)(filt_end - d_filteredKeys.begin());
    d_filteredKeys.resize(numFiltered);

    if (numFiltered == 0) {
        RCLCPP_WARN(rclcpp::get_logger("clustering_node"),
                     "No voxels survived countThreshold filter");
        return;
    }

    // ------------------------------------------------------------------
    // 6. Union-find on 26-connected voxel grid (GPU)
    // ------------------------------------------------------------------
    if ((int)d_parent.capacity() < numFiltered) d_parent.reserve(numFiltered);
    d_parent.resize(numFiltered);
    thrust::sequence(thrust::cuda::par(alloc).on(stream),
                     d_parent.begin(), d_parent.end());

    blocks = (numFiltered + threads - 1) / threads;
    unionFindKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_filteredKeys.data()),
        numFiltered,
        thrust::raw_pointer_cast(d_parent.data()),
        thrust::raw_pointer_cast(d_grid.data()));

    // ------------------------------------------------------------------
    // 7. Flatten parent array (GPU)
    // ------------------------------------------------------------------
    flattenParentKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_parent.data()), numFiltered);

    // ------------------------------------------------------------------
    // 8. Assign cluster label per point (GPU)
    // ------------------------------------------------------------------
    if (d_pointLabels.capacity() < inputSize) d_pointLabels.reserve(inputSize);
    d_pointLabels.resize(inputSize);

    blocks = (inputSize + threads - 1) / threads;
    assignClusterLabelsKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_voxelKeys.data()),
        inputSize,
        thrust::raw_pointer_cast(d_filteredKeys.data()),
        numFiltered,
        thrust::raw_pointer_cast(d_parent.data()),
        thrust::raw_pointer_cast(d_pointLabels.data()));

    // ------------------------------------------------------------------
    // 9. Build compact cluster IDs + per-cluster bbox (GPU)
    // ------------------------------------------------------------------
    // get unique root labels from parent array → compact cluster IDs
    if ((int)d_uniqueLabels.capacity() < numFiltered)
        d_uniqueLabels.reserve(numFiltered);
    d_uniqueLabels.resize(numFiltered);

    // copy parent, sort, unique to get distinct root labels
    thrust::copy(thrust::cuda::par(alloc).on(stream),
                 d_parent.begin(), d_parent.end(),
                 d_uniqueLabels.begin());
    thrust::sort(thrust::cuda::par(alloc).on(stream),
                 d_uniqueLabels.begin(), d_uniqueLabels.end());
    auto ul_end = thrust::unique(thrust::cuda::par(alloc).on(stream),
                                  d_uniqueLabels.begin(), d_uniqueLabels.end());
    int numClusters = (int)(ul_end - d_uniqueLabels.begin());
    d_uniqueLabels.resize(numClusters);

    if (numClusters == 0) {
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "No clusters found");
        return;
    }

    // build label → compact ID map (just sequential 0..numClusters-1)
    if ((int)d_labelMap.capacity() < numClusters) d_labelMap.reserve(numClusters);
    d_labelMap.resize(numClusters);
    thrust::sequence(thrust::cuda::par(alloc).on(stream),
                     d_labelMap.begin(), d_labelMap.end());

    // allocate and init per-cluster bbox + sizes
    if ((int)d_clusterBBox.capacity() < numClusters * 6)
        d_clusterBBox.reserve(numClusters * 6);
    d_clusterBBox.resize(numClusters * 6);

    if ((int)d_clusterSizes.capacity() < numClusters)
        d_clusterSizes.reserve(numClusters);
    d_clusterSizes.resize(numClusters);

    // init bbox: min = +inf, max = -inf ; sizes = 0
    thrust::fill(thrust::cuda::par(alloc).on(stream),
                 d_clusterSizes.begin(), d_clusterSizes.end(), 0u);
    {
        // init directly into d_clusterBBox — no temp device_vector
        float* raw_bbox = thrust::raw_pointer_cast(d_clusterBBox.data());
        thrust::for_each(thrust::cuda::par(alloc).on(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(numClusters * 6),
            [raw_bbox] __device__ (int i) {
                int channel = i % 6;
                raw_bbox[i] = (channel < 3) ? 1e30f : -1e30f;
            });
    }

    // per-cluster bbox accumulation
    blocks = (inputSize + threads - 1) / threads;
    clusterBBoxKernel<<<blocks, threads, 0, stream>>>(
        input,
        thrust::raw_pointer_cast(d_pointLabels.data()),
        thrust::raw_pointer_cast(d_labelMap.data()),
        numClusters,
        thrust::raw_pointer_cast(d_uniqueLabels.data()),
        inputSize,
        thrust::raw_pointer_cast(d_clusterBBox.data()),
        thrust::raw_pointer_cast(d_clusterSizes.data()));

    // ------------------------------------------------------------------
    // 10. Dimension filter on GPU → cone points
    // ------------------------------------------------------------------
    if ((int)d_conePoints.capacity() < numClusters * 3)
        d_conePoints.reserve(numClusters * 3);
    d_conePoints.resize(numClusters * 3);
    cudaMemsetAsync(thrust::raw_pointer_cast(d_numCones.data()), 0,
                    sizeof(unsigned int), stream);

    blocks = (numClusters + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    dimensionFilterKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_clusterBBox.data()),
        thrust::raw_pointer_cast(d_clusterSizes.data()),
        numClusters,
        ecp.minClusterSize, ecp.maxClusterSize,
        filterParams.clusterMinX, filterParams.clusterMinY, filterParams.clusterMinZ,
        filterParams.clusterMaxX, filterParams.clusterMaxY, filterParams.clusterMaxZ,
        filterParams.maxHeight,
        thrust::raw_pointer_cast(d_conePoints.data()),
        thrust::raw_pointer_cast(d_numCones.data()));

    // ------------------------------------------------------------------
    // 11. Single D→H sync: copy cone points (tiny — typically < 100 cones)
    // ------------------------------------------------------------------
    unsigned int numCones = 0;
    cudaMemcpyAsync(&numCones, thrust::raw_pointer_cast(d_numCones.data()),
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (numCones > 0) {
        std::vector<float> h_cones(numCones * 3);
        cudaMemcpyAsync(h_cones.data(), thrust::raw_pointer_cast(d_conePoints.data()),
                        numCones * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (unsigned int i = 0; i < numCones; ++i) {
            geometry_msgs::msg::Point pnt;
            pnt.x = h_cones[i * 3 + 0];
            pnt.y = h_cones[i * 3 + 1];
            pnt.z = h_cones[i * 3 + 2];
            cones->points.push_back(pnt);
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
    #ifdef ENABLE_VERBOSE
        totalTime += total_ms;
        iterations++;        
        std::cout << "From " << inputSize << " points → "
        << numVoxels << " voxels → "
        << numFiltered << " filtered → "
        << numClusters << " clusters → "
        << numCones << " cones\n"
        << "Clustering time: " << total_ms << " ms\n"
        << "Average time per iteration: " << totalTime / iterations << " ms after " << iterations << " iterations\n"
        << "-------------------------------------------------------" << std::endl;
    #else
        std::cout << "From " << inputSize << " points → "
        << numVoxels << " voxels → "
        << numFiltered << " filtered → "
        << numClusters << " clusters → "
        << numCones << " cones\n"
        << "Clustering time: " << total_ms << " ms\n"
        << "-------------------------------------------------------" << std::endl;
    #endif
}