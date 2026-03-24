#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

typedef struct
{
    double distanceThreshold;
    int maxIterations;
    double probability;
    bool optimizeCoefficients;
} segParam_t;

class Isegmentation
{
public:
    cudaStream_t stream = NULL;
    segParam_t segP;
    /**
     * → input point buffer (as floats xyzxyz...)
     * num_points: number of points
     * out_points: caller-allocated buffer, at least as big as input
     * out_num_points: actual number of points in segmentation result
     */
    virtual void segment(float *inputData,
                         int nCount,
                         float **out_points,
                         unsigned int* out_num_points) = 0;
};
