#include "PatchMatchGpu.h"
#include "CudaError.cuh"
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#define MAXCOST 10000
static __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x,
                       a.y - b.y,
                       a.z - b.z,
                       0);
}
__device__ float l1_norm(float4 f)
{
    return (fabsf(f.x) +
            fabsf(f.y) +
            fabsf(f.z)) *
           0.33333333f;
}

__device__ float l1_norm(float f)
{
    return fabsf(f);
}

__device__ void sort_small(float *d, const int n)
{
    int j;
    // 采用挖坑法进行排序
    for (int i = 1; i < n; ++i)
    {
        float tmp = d[i];
        // 相当于将大于tmp的数进行左移动，将tmp放到0位置或者第一个小于它的数右边
        for (j = i; j >= 1 && tmp < d[j - 1]; --j)
        {
            d[j] = d[j - 1];
        }
        d[j] = tmp;
    }
}

__device__ float curand_between(curandState *random, const float &minValue, const float &maxValue)
{
    return (curand_uniform(random) * (maxValue - minValue) + minValue);
}
__device__ void randomUnitVectorOnPlane_cu(float4 *norm, float4 &viewVector, curandState *random)
{
    // 随机产生平面法向量
    float x = 1.0f;
    float y = 1.0f;
    float sum = 2.0f;
    while (sum >= 1.0f)
    {
        x = curand_between(random, -1.0f, 1.0f);
        y = curand_between(random, -1.0f, 1.0f);
        sum = pow2(x) + pow2(y);
    }
    const float sq = sqrtf(sum);
    norm->x = 2.0f * x * sq;
    norm->y = 2.0f * y * sq;
    norm->z = 1.0f - 2.0f * sum;
    // 判断该法向量的方向是否值着光心方向
    const float dp = dot4((*norm), viewVector);
    if (dp > 0)
    {
        negate4(norm);
    }
    return;
}
__device__ void get3DPoint_cu(float4 *ptX, const Camera_cu &camera, const int2 &p)
{
    float4 pt;
    // k*(R*X+T)=x -> X=(K*R).inv()(x-k*T);
    pt.x = (float)p.x - camera.P_col34.x;
    pt.y = (float)p.y - camera.P_col34.y;
    pt.z = 1.0f - camera.P_col34.z;
    matvecmul4(camera.M_inv, pt, ptX);
}
__device__ void normalize_cu(float4 *v)
{
    const float normSquared = pow2(v->x) + pow2(v->y) + pow2(v->z);
    const float inverse_sqrt = rsqrt(normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
    v->z *= inverse_sqrt;
}
__device__ void getViewVector_cu(float4 *v, const Camera_cu &camera, const int2 &p)
{
    // 得到该点在世界坐标下的位置,假设深度为1
    get3DPoint_cu(v, camera, p);
    // 得到射线方向
    sub((*v), camera.C4);
    // 对射线进行归一化
    normalize_cu(v);
}

__device__ float disparityDepthConversion(const float &f, const float &baseline, const float &d)
{
    return f * baseline / d;
}

__device__ float getD_cu(const float4 &normal, const int2 &p, const float &depth, const Camera_cu &camera)
{
    float4 pt, ptX;
    pt.x = depth * (float)(p.x) - camera.P_col34.x;
    pt.y = depth * (float)(p.y) - camera.P_col34.y;
    pt.z = depth - camera.P_col34.z;
    matvecmul4(camera.M_inv, pt, (&ptX));
    // printf("p.x=%d,p.y=%d,depth=%d\n",pt.x,pt.y,depth);
    // printf("camera.P_col34.x=%d,P_col34.y=%d\n",camera.P_col34.x,camera.P_col34.y);
    // printf("ptX.x=%f,ptX.y=%f,ptX.z=%f\n",ptX.x,ptX.y,ptX.z);
    // printf("camera.M_inv=%f,%f,%f\n",camera.M_inv[0],camera.M_inv[1],camera.M_inv[2]);
    return -(dot4(normal, ptX));
}

__device__ void getHomography_cu(const Camera_cu &from,
                                 const Camera_cu &to,
                                 const float *k1_inv,
                                 const float *k2,
                                 const float4 &n,
                                 const float &d,
                                 float *H)
{
    float tmp2[16];
    // printf("to.t4:x=%f,y=%f,z=%f,w=%f\n",to.t4.x,to.t4.y,to.t4.z,to.t4.w);
    outer_product4(to.t4, n, H);
    matdivide(H, d);
    matmatsub2(to.R, H);
    matmul_cu(H, k1_inv, tmp2);
    // printf("k1_inv:fx=%f,u=%f,fy=%f,v=%f\n",k1_inv[0],k1_inv[2],k1_inv[4],k1_inv[5]);
    matmul_cu(k2, tmp2, H);
}
__device__ void getCorrrespondingPoint_cu(const int2 &p, const float *H, float4 *ptf)
{
    float4 pt;
    pt.x = __int2float_rn(p.x);
    pt.y = __int2float_rn(p.y);
    matvecmul4noz(H, pt, ptf);
    vecdiv4(ptf, ptf->z);
    return;
}
template <typename T>
__device__ float weight_cu(const T &c1, const T &c2, const float &gamma)
{
    const float colorDiff = l1_norm(c1 - c2);
    return expf(-colorDiff / gamma);
}
// z=(-dc)/((x-u,(y-v)*a,c)*n)
__device__ float getDepthFromPlane3_cu(const Camera_cu &camera, const float4 &norm, const float &d, const int2 &p)
{
    return -d * camera.fx / (norm.x * (p.x - camera.k[2]) + (norm.y * (p.y - camera.k[2 + 3])) * camera.alpha + norm.z * camera.fx);
}
__device__ float getDepth_cu(const float4 &normal, const float &d, const int2 &p, const Camera_cu &camera)
{
    return getDepthFromPlane3_cu(camera, normal, d, p);
}
template <typename T>
__device__ float pmCostComputation(const cudaTextureObject_t &l,
                                   const cudaTextureObject_t &r,
                                   const float4 &pt_l,
                                   const float4 &pt_r,
                                   const int &rows,
                                   const int &cols,
                                   const float &tau_color,
                                   const float &tau_gradient,
                                   const float &alpha,
                                   const float &w)
{
    
    // if (pt_l.x + 2 >= cols || pt_l.x  < 2 || pt_l.y + 2 >= rows || pt_l.y  < 2 || pt_r.x + 1 >= cols || pt_r.x  < 2 || pt_r.y + 1 >= rows || pt_r.y  < 2)
    //     return w * ((1.f - alpha) * tau_color + alpha * tau_gradient);
    //T colorDiff = tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 0.5f) - tex2D<T>(r, pt_r.x + 0.5f, pt_r.y + 0.5f);
    // printf("ok");
    //float4 rgb1=tex2D<float4>(l, pt_l.x + 0.5f, pt_l.y + 0.5f) - tex2D<float4>(r, pt_r.x + 0.5f, pt_r.y + 0.5f);
   
    // float4 rgb2=tex2D<float4>(r, pt_r.x + 0.5f, pt_r.y + 0.5f);
   
      //printf("rgb1.x=%f,rgb1.y=%f,rgb1.z=%f\n",rgb1.x,rgb1.y,rgb1.z);
    float colDiff = l1_norm(tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 0.5f) - tex2D<T>(r, pt_r.x + 0.5f, pt_r.y + 0.5f));
    // printf("colDiff=%f\n",colDiff);
    // printf("ok\n");
    const float col_dis = fminf(colDiff, tau_color);
    // if(col_dis<tau_color)
    //printf("col_dis=%f\n",col_dis);
    const T gx1 = tex2D<T>(l, pt_l.x + 1 + 0.5f, pt_l.y + 0.5f) - tex2D<T>(l, pt_l.x - 1 + 0.5f, pt_l.y + 0.5f);
    const T gy1 = tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 1 + 0.5f) - tex2D<T>(l, pt_l.x + 0.5f, pt_l.y - 1 + 0.5f);

    const T gx2 = tex2D<T>(r, pt_r.x + 1 + 0.5f, pt_r.y + 0.5f) - tex2D<T>(r, pt_r.x - 1 + 0.5f, pt_r.y + 0.5f);
    const T gy2 = tex2D<T>(r, pt_r.x + 0.5f, pt_r.y + 1 + 0.5f) - tex2D<T>(r, pt_r.x + 0.5f, pt_r.y - 1 + 0.5f);

    const T grad_x_diff = (gx1 - gx2);
    const T grad_y_diff = (gy1 - gy2);

    const float grad_dis = fminf((l1_norm(grad_x_diff) + l1_norm(grad_y_diff)) * 0.0625, tau_gradient);
    // printf("col_dis=%f,grad_dis=%f\n",col_dis,grad_dis);
    const float dis = (1.f - alpha) * col_dis + alpha * grad_dis;
    return w * dis;
}

template <typename T>
__device__ float pmCost(const cudaTextureObject_t &l,
                        const cudaTextureObject_t &r,
                        const int2 &p,
                        const float4 &normal,
                        const int &vRad,
                        const int &hRad,
                        const AlgorithmParameters &algParam,
                        const CameraParameters_cu &camParams,
                        const int &camTo,
                        const int reference)
{
    const int cols = camParams.cols;
    const int rows = camParams.rows;
    const float alpha = algParam.alpha;
    const float tau_color = algParam.tauColor;
    const float tau_gradient = algParam.tauGradient;
    const float gamma = algParam.gamma;

    float H[16];
    getHomography_cu(camParams.cameras[reference],
                     camParams.cameras[camTo],
                     camParams.cameras[reference].k_inv,
                     camParams.cameras[camTo].k,
                     normal,
                     normal.w,
                     H);
    float cost = 0;

    for (int i = -hRad; i < hRad + 1; ++i)
    {
        for (int j = -vRad; j < vRad + 1; ++j)
        {
            const int xTemp = p.x + i;
            const int yTemp = p.y + j;
            // if (xTemp >= cols || yTemp >= rows || xTemp < 0 || yTemp < 0)
            //     continue;
            float4 pt_l;
            pt_l.x = __int2float_rn(xTemp);
            pt_l.y = __int2float_rn(yTemp);
            int2 pt_li = make_int2(xTemp, yTemp);
            float w = weight_cu<T>(tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 0.5f), tex2D<T>(l, p.x + 0.5f, p.y + 0.5f), gamma);
          
            float4 pt_r;
            getCorrrespondingPoint_cu(pt_li, H, &pt_r);
           
            cost+=pmCostComputation<T>(l, r, pt_l, pt_r, rows, cols, tau_color, tau_gradient, alpha, w);
        
          
        }
    }
    // printf("cost=%f",cost);
    return cost;
}

template <typename T>
__device__ float pmCostMultiview_cu(const cudaTextureObject_t *images,
                                    const int2 &p,
                                    const float4 &norm,
                                    const int &vRad,
                                    const int &hRad,
                                    const AlgorithmParameters &algParam,
                                    const CameraParameters_cu &camParams,
                                    const float4 *state,
                                    const int reference)
{
    float costVector[32];
    float cost = 0.0f;
    int numValidViews = 0;
    int cost_count = 0;
    //  printf("idxCurr=%d", camParams.viewSelectionSubset[0]);
    // printf("normla.x%f,norm.y=%f,norm.z=%f,norm.w=%f",normal.x,normal.y,normal.z,normal.w);
    for (int i = 0; i < camParams.viewSelectionSubsetNumber; ++i)
    {
        int idxCurr = camParams.viewSelectionSubset[i];
        float c = pmCost<T>(images[reference],
                            images[idxCurr],
                            p,
                            norm,
                            vRad, hRad,
                            algParam, camParams,
                            idxCurr,
                            reference);
        // float c=0;
        //  printf("c=%f\n",c);
        // printf("reference=%d,indxCurr=%d\n",reference,idxCurr);
        if (c < MAXCOST)
        {
            
            numValidViews += 1;
        }
        else
        {
            c = MAXCOST;
        }
       
        costVector[i] = c;
        cost_count += 1;
    }
    sort_small(costVector, cost_count);
    float costThresh = costVector[0] * algParam.goodFactor;
    for (int i = 0; i < numValidViews; ++i)
    {
        cost = cost + fminf(costThresh, costVector[i]);
    }
    cost = cost / numValidViews;
    if (numValidViews < 1)
    {
        cost = MAXCOST;
    }
    if (cost != cost || cost > MAXCOST || cost < 0)
    {
        cost = MAXCOST;
    }
   
//    __syncthreads();
    //  cost = cost / numValidViews;
    // synchronized();
    // printf("cost=%f",cost);
    return cost;
}

template <typename T>
__global__ void initPlane_Cost(GlobalState &globalState)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int rows = globalState.cameras->rows;
    const int cols = globalState.cameras->cols;
    if (p.x >= cols || p.y >= rows)
    {
        return;
    }
    Camera_cu &camera = globalState.cameras->cameras[globalState.reference];
    const int center = p.y * cols + p.x;
    int winH_half = globalState.params->winHsize / 2;
    int winW_half = globalState.params->winWsize / 2;
    float disp_now;
    float4 norm_now;
    curandState localState = globalState.random[center];
    curand_init(clock64(), p.y, p.x, &localState);
    float minDisparity = globalState.params->minDisparity;
    float maxDisparity = globalState.params->maxDisparity;
    float4 viewVector;
    getViewVector_cu(&viewVector, camera, p);
    // 初始化随机视差
    disp_now = curand_between(&localState, minDisparity, maxDisparity);
    // 初始化平面法向量
    randomUnitVectorOnPlane_cu(&norm_now, viewVector, &localState);
    //printf("camera.baseline=%f\n",camera.baseline);
    float depth_now = disparityDepthConversion(camera.f, camera.baseline, disp_now);

    // 计算根据本质矩阵平面的定义 n*P=-d
    //   printf("p.x=%d,p.y=%d\n",p.x,p.y);
    norm_now.w = getD_cu(norm_now, p, depth_now, camera);
    // printf("norm4.x=%f,norm4.y=%f,norm4.z=%f,norm4.w=%f\n",norm_now.x,norm_now.y,norm_now.z,norm_now.w);
    // printf("globalState.lines->cost[center]=%f\n",globalState.lines->cost[center]);
    globalState.lines->cost[center] = pmCostMultiview_cu<T>(globalState.imgs,
                                                            p,
                                                            norm_now,
                                                            winH_half,
                                                            winW_half,
                                                            *(globalState.params),
                                                            *(globalState.cameras),
                                                            globalState.lines->norm4,
                                                            globalState.reference);
    globalState.lines->norm4[center] = norm_now;
    // printf("globalState.lines->cost[center]=%f\n",globalState.lines->cost[center]);
    return;
}
template <typename T>
__device__ void spatialPropagation(const cudaTextureObject_t *imgs,
                                   const int2 &pt,
                                   const int &winH_half, const int &winW_half,
                                   const AlgorithmParameters &algParams,
                                   const CameraParameters_cu &camParams,
                                   float *cost_now,
                                   float4 *norm_now,
                                   const float4& norm_before,
                                   float *depth_now,
                                   const float4 *norm_all,
                                   int reference)
{
    const float d_before = norm_before.w;
    const float depth_before = getDepth_cu(norm_before, d_before, pt, camParams.cameras[reference]);
    float cost_before = pmCostMultiview_cu<T>(imgs, pt, norm_before, winH_half, winW_half, algParams, camParams, norm_all, reference);
    if (depth_before > algParams.depthMin && depth_before < algParams.depthMax)
    {
        // printf("ok\n");
        if (cost_before < (*cost_now))
        {
            // printf("cost_before=%f",cost_before);
            *depth_now = depth_before,
            *norm_now = norm_before;
            *cost_now = cost_before;
        }
    }
    return;
}

template <typename T>
__device__ void SpatiolProp_cu(GlobalState &globalState, int2 &pt)
{
    const int rows = globalState.cameras->rows;
    const int cols = globalState.cameras->cols;
    if (pt.x >= cols || pt.y >= rows)
    {
        return;
    }
    int winH_half = (globalState.params->winHsize - 1) / 2;
    int winW_half = (globalState.params->winWsize - 1) / 2;
    LineState &line = (*globalState.lines);
    const int center = pt.y * cols + pt.x;

    AlgorithmParameters &algParams = *(globalState.params);
    CameraParameters_cu &camParams = *(globalState.cameras);
    const cudaTextureObject_t *imgs = globalState.imgs;
    float *cost = line.cost;
    float4 *norm = line.norm4;
    int reference = globalState.reference;
    float cost_now = cost[center];
    float4 norm_now = norm[center];
    float depth_now = getDepth_cu(norm_now, norm_now.w, pt, camParams.cameras[reference]);
    const int left1 = center - 1;
    const int right1 = center + 1;
    const int down1 = center + cols;
    const int up1 = center - cols;
    const int left5 = center - 5;
    const int right5 = center + 5;
    const int down5 = center + 5 * cols;
    const int up5 = center - 5 * cols;
    // const float cost_before=cost_now;
   
    if (pt.y > 0)
    { 
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[up1], &depth_now, norm, reference);
    }
     if (pt.y > 4)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[up5], &depth_now, norm, reference);
    }
    if (pt.y < rows - 1)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[down1], &depth_now, norm, reference);
    }
    if (pt.y < rows - 5)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[down5], &depth_now, norm, reference);
    }
    if (pt.x > 0)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[left1], &depth_now, norm, reference);
    }
    if (pt.x > 4)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[left5], &depth_now, norm, reference);
    }
    if (pt.x < cols - 1)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[right1], &depth_now, norm, reference);
    }
    if (pt.x < cols - 5)
    {
        spatialPropagation<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, norm[right5], &depth_now, norm, reference);
    }
    //printf("cost_now=%f,cost_before=%f\n",cost_now,cost_before);
    cost[center] = cost_now;
    norm[center] = norm_now;
    return;
}


template <typename T>
__global__ void Black_spatialProp_cu(GlobalState &globalState)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
   
    SpatiolProp_cu<T>(globalState, p);
}


template <typename T>
__global__ void Red_spatialProp_cu(GlobalState &globalState)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;
   
    SpatiolProp_cu<T>(globalState, p);
}
__device__ void getRndDisAndUnitVector_cu(float disp,
                                          const float4 norm,
                                          float &dispOut,
                                          float4 *normOut,
                                          const float maxDeltaZ,
                                          const float maxDeltaN,
                                          const float minDisparity,
                                          const float maxDisparity,
                                          curandState *random,
                                          CameraParameters_cu &camParams,
                                          const float baseline,
                                          const float4 viewVector)
{
    disp = disparityDepthConversion(camParams.f, baseline, disp);
    float minDelta, maxDelta;
    minDelta = -min(maxDeltaZ, minDisparity + disp);
    maxDelta = min(maxDeltaZ, maxDisparity - disp);
    float deltaZ = curand_between(random, minDelta, maxDelta);
    dispOut = fminf(fmaxf(disp + deltaZ, minDisparity), maxDisparity);
    dispOut = disparityDepthConversion(camParams.f, baseline, dispOut);
    normOut->x = norm.x + curand_between(random, -maxDeltaN, maxDeltaN);
    normOut->y = norm.y + curand_between(random, -maxDeltaN, maxDeltaN);
    normOut->z = norm.z + curand_between(random, -maxDeltaN, maxDeltaN);
    normalize_cu(normOut);
    if (dot4((*normOut), viewVector) > 0.0f)
    {
        negate4(normOut);
    }
    return;
}

template <typename T>
__device__ void Refinement_cu(const cudaTextureObject_t *imgs,
                              const int2 &pt,
                              const int winH_half, const int winW_half,
                              const AlgorithmParameters &algParams,
                              CameraParameters_cu &camParams,
                              float *cost_now,
                              float4 *norm_now,
                              float *depth_now,
                              curandState *random,
                              const float4 *norm_all,
                              const int reference)
{
    float deltaN = 1.0f;
    float4 viewVector;
    getViewVector_cu((&viewVector), camParams.cameras[reference], pt);
    float4 norm_temp;
    float depthTemp;
    float costTemp;
   
    for (float deltaZ = algParams.maxDisparity / 2.0f; deltaZ >= 0.01f; deltaZ /= 10.f)
    {
        getRndDisAndUnitVector_cu(*depth_now, *norm_now, depthTemp, &norm_temp, deltaZ, deltaN,
                                  algParams.maxDisparity, algParams.minDisparity, random, camParams,
                                  camParams.cameras[reference].baseline, viewVector);
        norm_temp.w = getD_cu(norm_temp, pt, depthTemp, camParams.cameras[reference]);
        //  printf("norm_temp.x=%f,norm_temp.y=%f,norm_temp.z=%f,norm_temp.w=%f\n",norm_temp.x,norm_temp.y,norm_temp.z,norm_temp.w);
        costTemp = pmCostMultiview_cu<T>(imgs, pt, norm_temp, winH_half, winW_half, algParams, camParams, norm_all, reference);
        if (costTemp < (*cost_now))
        {
            *cost_now = costTemp;
            *depth_now = depthTemp;
            *norm_now = norm_temp;
        }
        deltaN /= 4.0f;
    }
   
}

template <typename T>
__device__ void PlaneRefinement_cu(GlobalState &globalState, const int2 &pt)
{
    const int rows = globalState.cameras->rows;
    const int cols = globalState.cameras->cols;
    if (pt.x >= cols || pt.y >= rows)
    {
        return;
    }
    int winH_half = (globalState.params->winHsize - 1) / 2;
    int winW_half = (globalState.params->winWsize - 1) / 2;
    LineState &line = (*globalState.lines);
    const int center = pt.y * cols + pt.x;

    AlgorithmParameters &algParams = *(globalState.params);
    CameraParameters_cu &camParams = *(globalState.cameras);
    const cudaTextureObject_t *imgs = globalState.imgs;
    float *cost = line.cost;
    float4 *norm4 = line.norm4;
    int reference = globalState.reference;
    float cost_now = cost[center];
    float4 norm_now = norm4[center];
    float depth_now = getDepth_cu(norm_now, norm_now.w, pt, camParams.cameras[reference]);
    curandState localState = globalState.random[center];
    // printf("norm_temp.x=%f,norm_temp.y=%f,norm_temp.z=%f,norm_temp.w=%f\n",norm_now.x,norm_now.y,norm_now.z,norm_now.w);

    Refinement_cu<T>(imgs, pt, winH_half, winW_half, algParams, camParams, &cost_now, &norm_now, &depth_now, &localState, norm4, globalState.reference);
    cost[center] = cost_now;
    norm4[center] = norm_now;
    return;
}

template <typename T>
__global__ void Black_planeRefine_cu(GlobalState &globalstate)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;
    // printf("p.x=%d,p.y=%d",p.x,p.y);
    PlaneRefinement_cu<T>(globalstate, p);
}
template <typename T>
__global__ void Red_planeRefine_cu(GlobalState &globalstate)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;

    PlaneRefinement_cu<T>(globalstate, p);
}

__global__ void compute_Depth_cu(GlobalState &globalstate)
{
    const int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int rows = globalstate.params->rows;
    int cols = globalstate.params->cols;
    if (pt.x >= cols || pt.y >= rows)
    {
        return;
    }
    int reference = globalstate.reference;
    int center = pt.y * cols + pt.x;
    float4 norm = globalstate.lines->norm4[center];
    float4 world_norm;
    matvecmul4(globalstate.cameras->cameras[reference].R_orig_inv, norm, (&world_norm));
    if (globalstate.lines->cost[center] != MAXCOST)
    {
          world_norm.w = getDepth_cu(norm, norm.w, pt, globalstate.cameras->cameras[reference]);
    }
    else
    {
        world_norm.w = 0;
    }
    // printf("globalstate.lines->cost[center]=%f\n",globalstate.lines->cost[center]);
    // printf("world_norm.w=%f",world_norm.w);
    globalstate.lines->norm4[center] = world_norm;
}
template <typename T>
void patchMatch(GlobalState &globalState)
{
    int rows = globalState.cameras->rows;
    int cols = globalState.cameras->cols;
    // 创建随机种子
    checkCudaError(cudaMalloc(&globalState.random, rows * cols * sizeof(curandState)));
    int block_w = 32;
    int block_h = block_w / 2;
    dim3 block_size(block_w, block_h);
    dim3 grid_size((cols - 1) / block_w + 1, (rows / 2 - 1) / block_h + 1);
    dim3 block_size_init(16, 16);
    dim3 grid_size_init((cols - 1) / 16 + 1, (rows - 1) / 16 + 1);
    // 随机初始化平面法向量和代价
    initPlane_Cost<T><<<grid_size_init, block_size_init>>>(globalState);
    cudaDeviceSynchronize();
    int maxIteration = globalState.params->iterations;
    // 采用红黑块方法进行并行计算
    for (int i = 0; i < maxIteration; ++i)
    {
        Black_spatialProp_cu<T><<<grid_size,block_size>>>(globalState);
        cudaDeviceSynchronize();
        Black_planeRefine_cu<T><<<grid_size, block_size>>>(globalState);
        cudaDeviceSynchronize();
        Red_spatialProp_cu<T><<<grid_size,block_size>>>(globalState);
        cudaDeviceSynchronize();
        Red_planeRefine_cu<T><<<grid_size,block_size>>>(globalState);
        cudaDeviceSynchronize();
    }
    compute_Depth_cu<<<grid_size_init,block_size_init>>>(globalState);
    cudaDeviceSynchronize();
    cudaFree(&globalState.random);
    return;
}
void pathMatchStereo(int reference, GlobalState &globalState)
{
    

    if (globalState.params->colorProcessing)
    {
        patchMatch<float4>(globalState);
    }
    else
    {
        patchMatch<float>(globalState);
    }
    return;
}