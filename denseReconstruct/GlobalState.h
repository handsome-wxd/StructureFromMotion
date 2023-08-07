#ifndef GLOBALSTATE_H
#define GLOBALSTATE_H
#include "AlgorithmParameters.h"
#include "CameraParameter.h"
#include "LineState.h"
#include "CudaMemoryManaged.h"

#include<cuda_texture_types.h>
#include<curand_kernel.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<vector_types.h>
#include<driver_functions.h>
#include<cuda_gl_interop.h>
#include<builtin_types.h>

class GlobalState:public CudaMemoryManaged{
 public:
    CameraParameters_cu *cameras;
    LineState *lines;
    curandState *random;
    AlgorithmParameters* params;
    cudaTextureObject_t imgs[MAX_IMAGES];
    cudaArray *cuArray[MAX_IMAGES];
    int reference;
    GlobalState(){
        cameras=new CameraParameters_cu;
        lines=new LineState;
    }
    ~GlobalState(){
        delete cameras;
        delete lines;
        for(int i=0;i<MAX_IMAGES;++i){
            cudaFreeArray(cuArray[i]);
            cudaDestroyTextureObject(imgs[i]);
        }
    }
};
#endif