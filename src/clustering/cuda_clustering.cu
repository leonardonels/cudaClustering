#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/clustering/cluster_filtering/dimension_filter.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// ==========================================================================
//  KERNEL 1:  Compute a voxel hash for every point
// ==========================================================================
//  Hash = ix + iy * GRID + iz * GRID * GRID
//  where ix = floor((x - minX) / voxelX), etc.
//  Points with fewer than countThreshold neighbours in the same voxel
//  will be filtered later on the host side.
// ==========================================================================
__global__ void computeVoxelKeysKernel(
    const float* __restrict__ points,
    int* __restrict__ keys,
    unsigned int nPoints,
    float voxelX, float voxelY, float voxelZ,
    float minX, float minY, float minZ,
    int gridX, int gridY)
{
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
//  KERNEL 2:  Union-find on voxel grid — 26-connectivity
// ==========================================================================
//  Each thread handles one occupied voxel and merges it with all occupied
//  neighbours.  The voxel→linear-id mapping is passed via a hash table
//  stored in global memory (sorted arrays for binary search).
// ==========================================================================
__global__ void unionFindKernel(
    const int* __restrict__ uniqueKeys,   // sorted unique voxel hashes
    int  numVoxels,
    int* __restrict__ parent,
    int gridX, int gridY)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)numVoxels) return;

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
//  KERNEL 3:  Flatten parent array (path compress to root)
// ==========================================================================
__global__ void flattenParentKernel(int* parent, int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)n) return;
    parent[tid] = uf_find(parent, tid);
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

    filter = new DimensionFilter(param.filtering);
    cudaStreamCreate(&stream);
}

CudaClustering::~CudaClustering()
{
    if (stream != NULL) cudaStreamDestroy(stream);
    delete filter;
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
//  extractClusters — full custom CUDA pipeline
// --------------------------------------------------------------------------
//  1. Compute voxel hash per point                (GPU)
//  2. Sort points by voxel hash                   (GPU — thrust)
//  3. Reduce to unique voxels + per-voxel counts  (GPU — thrust)
//  4. Filter voxels by countThreshold             (CPU — small)
//  5. Union-find 26-connectivity                  (GPU)
//  6. Flatten labels, gather clusters             (CPU — small)
//  7. Per-cluster dimension filter → cones        (CPU)
// --------------------------------------------------------------------------
void CudaClustering::extractClusters(
    float* input,             // device pointer  (x,y,z,i)*N
    unsigned int inputSize,
    float* outputEC,          // device pointer  (pre-allocated, same size)
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

    // ------------------------------------------------------------------
    // Find bounding box on CPU (tiny copy — 6 floats via thrust)
    // ------------------------------------------------------------------
    // We wrap the device pointer in a thrust device_ptr so we can use
    // thrust algorithms without extra copies.
    thrust::device_ptr<float> dp(input);

    // Strided min/max — extract X/Y/Z channels
    // For simplicity and reliability we copy the points to host once.
    // The copy is overlapped with the rest of the pipeline anyway.
    std::vector<float> h_points(inputSize * 4);
    cudaMemcpyAsync(h_points.data(), input, inputSize * 4 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float minX =  1e30f, minY =  1e30f, minZ =  1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    for (unsigned int i = 0; i < inputSize; ++i) {
        float x = h_points[i*4+0], y = h_points[i*4+1], z = h_points[i*4+2];
        if (x < minX) minX = x;  if (x > maxX) maxX = x;
        if (y < minY) minY = y;  if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;  if (z > maxZ) maxZ = z;
    }

    // Grid dimensions
    int gridX = (int)ceilf((maxX - minX) / ecp.voxelX) + 1;
    int gridY = (int)ceilf((maxY - minY) / ecp.voxelY) + 1;
    int gridZ = (int)ceilf((maxZ - minZ) / ecp.voxelZ) + 1;
    (void)gridZ; // used implicitly in hash

    // ------------------------------------------------------------------
    // Compute voxel hash per point (GPU)
    // ------------------------------------------------------------------
    if (d_voxelKeys.capacity() < inputSize) {
        d_voxelKeys.reserve(inputSize);
        d_sortedKeys.reserve(inputSize);
        d_sortedIndices.reserve(inputSize);
    }
    d_voxelKeys.resize(inputSize);
    d_sortedKeys.resize(inputSize);
    d_sortedIndices.resize(inputSize);

    int blocks = (inputSize + threads - 1) / threads;
    computeVoxelKeysKernel<<<blocks, threads, 0, stream>>>(
        input, thrust::raw_pointer_cast(d_voxelKeys.data()), inputSize,
        ecp.voxelX, ecp.voxelY, ecp.voxelZ,
        minX, minY, minZ, gridX, gridY);

    // ------------------------------------------------------------------
    // Sort by voxel key (GPU)
    // ------------------------------------------------------------------
    d_sortedKeys = d_voxelKeys;
    thrust::sequence(thrust::cuda::par.on(stream),
                     d_sortedIndices.begin(), d_sortedIndices.end());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        d_sortedKeys.begin(), d_sortedKeys.end(),
                        d_sortedIndices.begin());

    // ------------------------------------------------------------------
    // Reduce to unique voxels + counts (GPU → host)
    // ------------------------------------------------------------------
    // Upper-bound on unique voxels = inputSize
    if (d_uniqueKeys.capacity() < inputSize)  d_uniqueKeys.reserve(inputSize);
    if (d_voxelCounts.capacity() < inputSize) d_voxelCounts.reserve(inputSize);
    d_uniqueKeys.resize(inputSize);
    d_voxelCounts.resize(inputSize);

    auto new_end = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        d_sortedKeys.begin(), d_sortedKeys.end(),
        thrust::make_constant_iterator(1u),
        d_uniqueKeys.begin(),
        d_voxelCounts.begin());

    int numVoxels = (int)(new_end.first - d_uniqueKeys.begin());
    d_uniqueKeys.resize(numVoxels);
    d_voxelCounts.resize(numVoxels);

    // Copy to host for filtering by countThreshold
    thrust::host_vector<int>          h_uniqueKeys  = d_uniqueKeys;
    thrust::host_vector<unsigned int> h_voxelCounts = d_voxelCounts;

    // Filter out voxels below countThreshold — build mapping
    std::vector<int>          filteredKeys;
    std::vector<unsigned int> filteredCounts;
    filteredKeys.reserve(numVoxels);
    filteredCounts.reserve(numVoxels);
    for (int i = 0; i < numVoxels; ++i) {
        if ((int)h_voxelCounts[i] >= ecp.countThreshold) {
            filteredKeys.push_back(h_uniqueKeys[i]);
            filteredCounts.push_back(h_voxelCounts[i]);
        }
    }
    int numFiltered = (int)filteredKeys.size();
    if (numFiltered == 0) {
        RCLCPP_WARN(rclcpp::get_logger("clustering_node"),
                     "No voxels survived countThreshold filter");
        return;
    }

    // Upload filtered keys for union-find
    d_uniqueKeys.assign(filteredKeys.begin(), filteredKeys.end());

    // ------------------------------------------------------------------
    // Union-find on 26-connected voxel grid (GPU)
    // ------------------------------------------------------------------
    if ((int)d_parent.capacity() < numFiltered) d_parent.reserve(numFiltered);
    d_parent.resize(numFiltered);
    thrust::sequence(thrust::cuda::par.on(stream),
                     d_parent.begin(), d_parent.end());

    blocks = (numFiltered + threads - 1) / threads;
    unionFindKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_uniqueKeys.data()),
        numFiltered,
        thrust::raw_pointer_cast(d_parent.data()),
        gridX, gridY);

    flattenParentKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_parent.data()), numFiltered);

    // Copy labels to host
    thrust::host_vector<int> h_parent = d_parent;

    // ------------------------------------------------------------------
    // Group voxels by cluster label (CPU)
    // ------------------------------------------------------------------
    // Map: root label → cluster id
    std::unordered_map<int, int> rootToCluster;
    int nextCluster = 0;
    for (int i = 0; i < numFiltered; ++i) {
        int root = h_parent[i];
        if (rootToCluster.find(root) == rootToCluster.end()) {
            rootToCluster[root] = nextCluster++;
        }
    }

    // Build per-cluster point lists using the sorted key arrays on host
    // We need sorted keys and sorted indices on host
    thrust::host_vector<int>          h_sortedKeys    = d_sortedKeys;
    thrust::host_vector<unsigned int> h_sortedIndices = d_sortedIndices;

    // Build a lookup: filteredKey → filtered-index (for parent lookup)
    std::unordered_map<int, int> keyToFilteredIdx;
    for (int i = 0; i < numFiltered; ++i) {
        keyToFilteredIdx[filteredKeys[i]] = i;
    }

    // Collect points per cluster
    // clusterPoints[clusterID] = list of original point indices
    std::vector<std::vector<unsigned int>> clusterPoints(nextCluster);
    for (unsigned int i = 0; i < inputSize; ++i) {
        int key = h_sortedKeys[i];
        auto it = keyToFilteredIdx.find(key);
        if (it == keyToFilteredIdx.end()) continue;  // voxel was filtered out
        int filtIdx = it->second;
        int root    = h_parent[filtIdx];
        int cid     = rootToCluster[root];
        clusterPoints[cid].push_back(h_sortedIndices[i]);
    }

    // ------------------------------------------------------------------
    // Filter clusters by size, write output, run dimension filter
    // ------------------------------------------------------------------
    // Write clusters sequentially into outputEC on device.
    // Build indexEC compatible with the old format:
    //   indexEC[0] = numClusters
    //   indexEC[i] = number of points in cluster i  (1-based)

    auto t2 = std::chrono::steady_clock::now();

    unsigned int totalOut = 0;
    std::vector<unsigned int> validClusterSizes;
    std::vector<unsigned int> validClusterOffsets;

    for (int c = 0; c < nextCluster; ++c) {
        unsigned int sz = (unsigned int)clusterPoints[c].size();
        if (sz < ecp.minClusterSize || sz > ecp.maxClusterSize) continue;
        validClusterOffsets.push_back(totalOut);
        validClusterSizes.push_back(sz);
        totalOut += sz;
    }

    if (totalOut == 0) {
        RCLCPP_INFO(rclcpp::get_logger("clustering_node"),
                     "No clusters survived size filter [%u, %u]",
                     ecp.minClusterSize, ecp.maxClusterSize);
        return;
    }

    // Build a flat host buffer with the clustered points
    std::vector<float> h_output(totalOut * 4);
    unsigned int writeIdx = 0;
    unsigned int finalCluster = 0;

    for (int c = 0; c < nextCluster; ++c) {
        unsigned int sz = (unsigned int)clusterPoints[c].size();
        if (sz < ecp.minClusterSize || sz > ecp.maxClusterSize) continue;
        for (unsigned int pidx : clusterPoints[c]) {
            h_output[writeIdx * 4 + 0] = h_points[pidx * 4 + 0];
            h_output[writeIdx * 4 + 1] = h_points[pidx * 4 + 1];
            h_output[writeIdx * 4 + 2] = h_points[pidx * 4 + 2];
            h_output[writeIdx * 4 + 3] = h_points[pidx * 4 + 3];
            writeIdx++;
        }
        finalCluster++;
    }

    // Copy clustered points to device output
    cudaMemcpyAsync(outputEC, h_output.data(), totalOut * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // ------------------------------------------------------------------
    // Per-cluster dimension filter → cones (CPU, uses host data)
    // ------------------------------------------------------------------
    for (unsigned int i = 0; i < finalCluster; ++i) {
        unsigned int offset = validClusterOffsets[i];
        unsigned int sz     = validClusterSizes[i];

        std::optional<geometry_msgs::msg::Point> pnt_opt =
            filter->analiseCluster(&h_output[offset * 4], sz);

        if (pnt_opt.has_value()) {
            cones->points.push_back(pnt_opt.value());
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    auto gpu_ms   = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t3 - t1).count();
    totalTime += total_ms;
    iterations++;

    std::cout << "From " << inputSize << " points to " 
              << numFiltered << " voxels to " 
              << finalCluster << " clusters to " 
              << totalOut << " output points\n"
              << "Clustering time: " << gpu_ms << " ms (GPU), " << total_ms << " ms (total)\n"
              << "Avarage time per iteration: " << totalTime / iterations << " ms after " << iterations << " iterations\n"
              << "-------------------------------------------------------" << std::endl;
}
