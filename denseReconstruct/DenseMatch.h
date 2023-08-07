#ifndef DENSEMATCH_H
#define DENSEMATCH_H
#include "AlgorithmParameters.h"
#include "../base/Image.h"
#include "GlobalState.h"

#include "AlgorithmParameters.h"
#include<vector>
#include "UtilGpu.h"
#include "PatchMatchGpu.h"

void PreconditioningGPUdata(std::vector<ImagePtr>&Images_,const int reference);
void patchMatch();
void DenseMatch(std::vector<ImagePtr>&Images_);
void selectViews(CameraParameters& camera,int imgWidth,int imgHight,AlgorithmParameters& algParams);
CameraParameters setCameraParameterscu(CameraParameters_cu& CP,std::vector<ImagePtr>& Images,const int reference,float scaleFactor=1.0f,bool transformP=true);
void addImageToTextureFloatGray (std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[]);
void addImageToTextureFloatColor(std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[]);
#endif