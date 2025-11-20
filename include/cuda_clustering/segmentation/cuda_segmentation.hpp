#pragma once

#include "isegmentation.hpp"

typedef enum
{
    SACMODEL_PLANE = 0,
    SACMODEL_LINE,
    SACMODEL_CIRCLE2D,
    SACMODEL_CIRCLE3D,
    SACMODEL_SPHERE,
    SACMODEL_CYLINDER,
    SACMODEL_CONE,
    SACMODEL_TORUS,
    SACMODEL_PARALLEL_LINE,
    SACMODEL_PERPENDICULAR_PLANE,
    SACMODEL_PARALLEL_LINES,
    SACMODEL_NORMAL_PLANE,
    SACMODEL_NORMAL_SPHERE,
    SACMODEL_REGISTRATION,
    SACMODEL_REGISTRATION_2D,
    SACMODEL_PARALLEL_PLANE,
    SACMODEL_NORMAL_PARALLEL_PLANE,
    SACMODEL_STICK,
} SacModel;

typedef enum
{
    SAC_RANSAC = 0,
    SAC_LMEDS = 1,
    SAC_MSAC = 2,
    SAC_RRANSAC = 3,
    SAC_RMSAC = 4,
    SAC_MLESAC = 5,
    SAC_PROSAC = 6,
} SacMethod;

typedef struct
{
    double distanceThreshold;
    int maxIterations;
    double probability;
    bool optimizeCoefficients;
} segParam_t;

class cudaSegmentation
{
public:
    // Now Just support: SAC_RANSAC + SACMODEL_PLANE
    cudaSegmentation(int ModelType, int MethodType, cudaStream_t stream = 0);

    ~cudaSegmentation(void);

    /*
    Input:
        cloud_in: data pointer for points cloud
        nCount: count of points in cloud_in
    Output:
        Index: data pointer which has the index of points in a plane from input
        modelCoefficients: data pointer which has the group of coefficients of the plane
    */
    int set(segParam_t param);
    void segment(float *cloud_in, int nCount,
                 int *index, float *modelCoefficients);

private:
};

class CudaSegmentation : public Isegmentation
{

private:
public:
    // unsigned int memory_allocated = 0;
    float *input = nullptr;
    int *index = nullptr;
    float *modelCoefficients = nullptr;
    cudaStream_t stream = NULL;
    segParam_t segP;
    bool skip = false;
    int mall_size = 0;
    CudaSegmentation(segParam_t& params);
    // void reallocateMemory(unsigned int size);
    void freeResources();
    void realloc(unsigned int size);
    void segment(float *inputData,
                 int nCount,
                 float **out_points,
                 unsigned int *out_num_points) override;
};