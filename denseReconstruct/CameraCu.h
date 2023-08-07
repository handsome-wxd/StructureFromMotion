#ifndef CAMERACU_H
#define CAMERACU_H
#include "CudaMemoryManaged.h"
#include "CudaError.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

class Camera_cu:public CudaMemoryManaged{
public:
    float* P;
    float4 P_col34;
    float* P_inv;
    float* M_inv;
    float* R;
    float* R_orig_inv;
    float4 t4;
    float4 C4;
    float fx;
    float fy;
    float f;
    float alpha;
    float baseline;
    bool reference;
    float depthMin;
    float depthMax;
    char* id;
    float* k;
    float* k_inv;
    Camera_cu(){
        baseline=1.0f;
        reference=false;
        depthMin=2.0f;
        depthMax=20.0f;
        checkCudaError(cudaMallocManaged(&P,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&P_inv,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&M_inv,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&k,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&k_inv,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&R,sizeof(float)*4*4));
        checkCudaError(cudaMallocManaged(&R_orig_inv,sizeof(float)*4*4));
    }
    ~Camera_cu(){
        cudaFree(P);
        cudaFree(P_inv);
        cudaFree(M_inv);
        cudaFree(k);
        cudaFree(k_inv);
        cudaFree(R);
        cudaFree(R_orig_inv);
    }
};
#endif