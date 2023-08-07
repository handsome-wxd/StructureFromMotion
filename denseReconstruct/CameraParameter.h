#ifndef CAMERAPARAMETER_H
#define CAMERAPARAMETER_H
#include "CameraCu.h"
#include "CudaMemoryManaged.h"
#include "UtilGpu.h"
#include <opencv2/opencv.hpp>

class __align__(128) CameraParameters_cu: public CudaMemoryManaged{
    public:
        float f;
        bool rectified;
        Camera_cu cameras[MAX_IMAGES];
        int idRef;
        int cols;
        int rows;
        int* viewSelectionSubset;
        int viewSelectionSubsetNumber;
        CameraParameters_cu(){
            rectified=true;
            idRef=0;
            checkCudaError(cudaMallocManaged(&viewSelectionSubset,sizeof(int)*MAX_IMAGES));
        }
        ~CameraParameters_cu(){
           cudaFree(viewSelectionSubset);
        }
};
struct Camera
{
    Camera():
    P(cv::Mat::eye(3,4,CV_32F)),
    R(cv::Mat::eye(3,3,CV_32F)),
    baseline(0.54f),
    reference(false),
    depthMin(2.0f),
    depthMax(20.0f){}
    cv::Mat_<float>P;
    cv::Mat_<float>P_inv;
    cv::Mat_<float>M_inv;
    cv::Mat_<float>R;
    cv::Mat_<float>R_orig_inv;
    cv::Mat_<float>t;
    cv::Vec3f Center;
    float baseline;
    bool reference;
    float depthMin;
    float depthMax;
    std::string id;
    cv::Mat_<float>k;
    cv::Mat_<float>k_inv;
};

struct CameraParameters{
    CameraParameters():rectified(false),idRef(0){}
    cv::Mat_<float>k;
    cv::Mat_<float>k_inv;
    float f;
    bool rectified;
    std::vector<Camera>cameras;
    int idRef;
    std::vector<int>viewSelectionSubset;
};
#endif