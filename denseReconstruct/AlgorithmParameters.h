#ifndef ALGORITHMPARAMETERS_H
#define ALGORITHMPARAMETERS_H
#include "CudaMemoryManaged.h"
#include<iostream>
struct AlgorithmParameters:public CudaMemoryManaged
{
    /* data */
    AlgorithmParameters():
    algorithm(0),
    maxDisparity(128.0f),
    minDisparity(-128.0f),
    winHsize(35),
    winWsize(35),
    tauColor(10),
    tauGradient(2.0f),
    alpha(0.9f),
    gamma(10.0f),
    borderValue(-1),
    iterations(3),
    colorProcessing(true),
    dispTol(1.0f),
    normTol(0.1f),
    censusEpsilon(2.5f),
    selfSimilarityN(50),
    camScale(1.0f),
    numImgProcessed(1),
    costThresh(40.f),
    goodFactor(1.5f),
    nBest(2),
    costComb(1),
    viewSelection(true),
    depthMin(-1.0f),
    depthMax(-1.0f),
    minAngle(5.0f),
    maxAngle(45.0f),
    baseline(1.0f),
    maxViews(9){}
    int algorithm;
    float maxDisparity;
    float minDisparity;
    int winHsize;
    int winWsize;
    float tauColor;
    float tauGradient;
    float alpha;
    float gamma;
    int borderValue;
    int iterations;
    bool colorProcessing;
    float dispTol;
    float  normTol;
    float censusEpsilon;
    int selfSimilarityN;
    float camScale;
    int numImgProcessed;
    float costThresh;
    float goodFactor;
    int nBest;
    int costComb;
    bool viewSelection;
    float depthMin;
    float depthMax;
    float minAngle;
    float maxAngle;
    unsigned int maxViews;
    float baseline;
    int cols;
    int rows;
   

};


#endif