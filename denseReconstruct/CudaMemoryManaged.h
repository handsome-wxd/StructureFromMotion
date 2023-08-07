#ifndef MANAGED_H
#define MANAGED_H
#include <cstddef>
#include "CudaError.cuh"
class CudaMemoryManaged{
    public:
        void *operator new(size_t len){
            void *ptr;
           
            checkCudaError(cudaMallocManaged(&ptr,len));
            return ptr;
        }
        void operator delete(void *ptr){
            cudaFree(ptr);
        }
};
#endif