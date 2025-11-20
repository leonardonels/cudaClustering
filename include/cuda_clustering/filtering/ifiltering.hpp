#pragma once 

class IFilter
{
    protected:
        
    public:
        virtual void filterPoints(float* input, unsigned int inputSize, float** output, unsigned int* outputSize) = 0;
};