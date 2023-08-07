#ifndef LINESTATE_H
#define LINESTATE_H
#include <string.h>
#include "AlgorithmParameters.h"
#include "CameraParameter.h"
#include "CudaMemoryManaged.h"
#include <vector_types.h>
class __align__(128) LineState:public CudaMemoryManaged{
 public:
    float4* norm4;
    float* cost;
    int n;
    int stride;
    int length;
    void mallocSpace(size_t n){
        checkCudaError(cudaMallocManaged(&cost,sizeof(float)*n));
        checkCudaError(cudaMallocManaged(&norm4,sizeof(float4)*n));
        memset(cost,0,sizeof(float)*n);
        memset(norm4,0,sizeof(float4)*n);
    }
    ~LineState(){
        cudaFree(cost);
        cudaFree(norm4);
    }
};
#endif