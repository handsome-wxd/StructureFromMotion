#ifndef UTILGPU_H
#define UTILGPU_H
#include"iostream"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "CudaError.cuh"
#define MAX_IMAGES 10
void selectCudaDevice();

#endif 