#pragma once

#include "isegmentation.hpp"
#include <cuda_runtime.h>

class CudaSegmentationKernel : public Isegmentation
{
public:
    int device_id = 0;

    CudaSegmentationKernel(segParam_t& params);
    ~CudaSegmentationKernel();

    void segment(float *inputData,
                 int nCount,
                 float **out_points,
                 unsigned int *out_num_points) override;
};