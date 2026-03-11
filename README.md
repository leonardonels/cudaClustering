# CUDA Clustering Node

A high-performance ROS2 node for real-time LiDAR point cloud processing, running the **entire pipeline on the GPU** using custom CUDA kernels. Designed for low-latency autonomous systems on both x86 and NVIDIA Jetson platforms.

## Pipeline

```
PointCloud2 → Passthrough Filter → RANSAC Segmentation → Voxel Clustering → Cone Markers
```

Each stage operates on `thrust::device_vector` GPU buffers — data stays on the GPU between stages with no unnecessary device-to-host copies.

## Features

### Passthrough Filter (`cuda_filtering`)
- Single-pass CUDA kernel checking X, Y, Z bounds simultaneously
- Atomic compaction — no thrust intermediate buffers
- Per-axis enable/disable via config

### RANSAC Ground Segmentation (`cuda_segmentation`)
- **Fully parallel GPU RANSAC** — all iterations run as concurrent CUDA blocks
- `wang_hash` PRNG for unbiased seed point selection
- Shared-memory inlier count reduction per block
- Best-plane selection via `thrust::max_element` (no D→H sync)
- Inlier marking reads winning plane directly from device memory

### Voxel-Based Euclidean Clustering (`cuda_clustering`)
7 custom CUDA kernels forming a fully GPU-resident pipeline:

1. **Bounding box** — shared-memory reduction with `atomicMinFloat`/`atomicMaxFloat`
2. **Voxel hashing** — reads bbox from device, writes grid dims to device
3. **Sort + reduce** — `thrust::sort` + `thrust::reduce_by_key` (in-place, cached allocator)
4. **Voxel count filter** — `thrust::copy_if` removes sparse voxels
5. **Union-find clustering** — 26-connectivity on voxel grid, lock-free `atomicCAS`
6. **Per-point label assignment** — binary search in filtered keys
7. **Per-cluster bbox + dimension filter** — atomic bbox accumulation, cone detection

Single D→H sync at the end: only the cone center points (typically < 100 floats) are copied to the host.

### Performance Optimizations
- **CachedAllocator** — reusable device memory pool for all thrust temporary buffers, eliminates per-frame `cudaMalloc`/`cudaFree` overhead (critical on Jetson's unified memory allocator)
- **Pinned host memory** — auto-enabled on x86 (`USE_PINNED_MEMORY`), uses `thrust::pinned_allocator` for the host input buffer to enable true async DMA transfers
- **Zero-copy pipeline** — `d_input.swap(d_output)` between stages, no intermediate host copies
- **Pre-sized buffers** — all device vectors are reserved to max capacity on first frame; zero allocations during steady-state
- **All logging behind `ENABLE_VERBOSE`** — no `std::cout` overhead in production builds

## Requirements

- **ROS2 Humble**
- **CUDA Toolkit** (≥ 11.0)
- **PCL** (Point Cloud Library)
- **Thrust** (included with CUDA toolkit)

### Supported Platforms
| Platform | Architecture | Pinned Memory | Notes |
|----------|-------------|---------------|-------|
| x86_64 desktop/server | `x86_64` | Enabled by default | Discrete GPU |
| NVIDIA Jetson Orin | `aarch64` | Disabled by default | Unified memory — pinned allocation is a no-op |

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select clustering
source install/setup.bash
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_PINNED_MEMORY` | `ON` (x86), `OFF` (ARM) | Use CUDA pinned host memory for H→D transfers |
| `ENABLE_VERBOSE` | `OFF` | Print essential timings information for each callback |

Example with options:
```bash
colcon build --packages-select clustering --cmake-args -DENABLE_VERBOSE=ON
```

## Configuration

All parameters are set in `config/config.yaml`:

```yaml
/clustering_node:
  ros__parameters:
    # Topics
    input_topic: /lidar_points
    segmented_topic: /segmented_points
    filtered_topic: /filtered_points
    cluster_topic: /clusters
    frame_id: "hesai_lidar"

    # Clustering
    minClusterSize: 1           # Min points per cluster
    maxClusterSize: 500         # Max points per cluster
    voxelX: 0.8                 # Voxel size X (meters)
    voxelY: 0.8                 # Voxel size Y (meters)
    voxelZ: 0.8                 # Voxel size Z (meters)
    countThreshold: 5           # Min points per voxel to keep

    # Cluster dimension filter (cone detection)
    clusterMaxX: 0.4            # Max cluster width X
    clusterMaxY: 0.4            # Max cluster width Y
    clusterMaxZ: 0.4            # Max cluster height
    clusterMinX: -0.1           # Min cluster width X
    clusterMinY: -0.1           # Min cluster width Y
    clusterMinZ: 0.1            # Min cluster height
    maxHeight: 0.4              # Max Z of lowest point

    # Passthrough filter bounds
    downFilterLimitX: -50.0
    upFilterLimitX: 50.0
    downFilterLimitY: -50.0
    upFilterLimitY: 50.0
    downFilterLimitZ: -5.0
    upFilterLimitZ: 5.0

    # RANSAC segmentation
    distanceThreshold: 0.1      # Inlier distance to plane (meters)
    maxIterations: 80           # RANSAC iterations (= CUDA blocks, 8 run in parallel on Orin)
    probability: 0.75

    # Pipeline stage enables
    filter: false               # Passthrough XYZ filter
    publishFilteredPc: false    # Publish filtered point cloud
    segment: true               # RANSAC ground removal
    publishSegmentedPc: false   # Publish segmented point cloud
    clustering: true            # Voxel clustering
    publishCluster: true        # Publish cone markers
```

### RANSAC `maxIterations` Tuning

Each iteration runs as a separate CUDA block with 1024 threads. On Jetson Orin (16 SMs, 2048 cuda):
- **16 blocks run truly in parallel** (1 block/SM at 1024 threads)
- Remaining blocks are queued and pipelined efficiently
- `maxIterations = 160` → ~10 sequential waves → good balance of robustness vs speed
- Range `130-160` is optimal for ground plane detection on filtered point clouds

## Run

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch clustering cuda_clustering_launch.py
```

## ROS2 Interface

### Subscriptions
| Topic | Type | Description |
|-------|------|-------------|
| `input_topic` | `sensor_msgs/PointCloud2` | Input LiDAR point cloud |

### Publications
| Topic | Type | Condition | Description |
|-------|------|-----------|-------------|
| `filtered_topic` | `sensor_msgs/PointCloud2` | `publishFilteredPc: true` | Filtered point cloud |
| `segmented_topic` | `sensor_msgs/PointCloud2` | `publishSegmentedPc: true` | Ground-removed point cloud |
| `cluster_topic` | `visualization_msgs/Marker` | `publishCluster: true` | Detected cone center points |

## Architecture

```
src/
├── main.cpp                           # Entry point
├── controller_node.cu                 # ROS2 node, pipeline orchestration
├── filtering/
│   └── cuda_filtering.cu             # Passthrough filter kernel
├── segmentation/
│   └── cuda_segmentation.cu          # RANSAC kernels (plane fit + inlier mark + compact)
├── clustering/
│   └── cuda_clustering.cu            # 7 clustering kernels (bbox → voxel → union-find → filter)
└── utils/
    └── pointcloud_converter.cpp       # PointCloud2 → float array (optimized with -O3)

include/cuda_clustering/
├── controller_node.hpp
├── utils/
│   ├── pointcloud_converter.hpp
│   └── cached_allocator.hpp           # Reusable device memory pool for thrust
├── filtering/
│   ├── ifiltering.hpp
│   └── cuda_filtering.hpp
├── segmentation/
│   ├── isegmentation.hpp
│   └── cuda_segmentation.hpp
└── clustering/
    ├── iclustering.hpp
    └── cuda_clustering.hpp
```

## License

Apache-2.0
