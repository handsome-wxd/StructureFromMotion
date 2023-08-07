#ifndef FILECONFIG_H
#define FILECONIFG_H
#include "baseH.h"



class Config{
public:
    //数据读取
    std::string ImagesPath;
    std::string IntrinsicsPath;
    bool isEvalIntrisic;
    //特征匹配
    std::string ExtractorName;
    int numFeatures;
    bool saveMatch;
    float featureThreshold;
    int flannK;
    int matchThreadNum;
    int matchRequestNum;
    //稀疏重建
    int trackTheshold;
    int edgeThreshold;
    int candiateImagePair;
    int minAngleTheshold;
    int maxAngleTheshold;
    std::string savePointCloudPath;
    std::string savePointPath;
    std::string savePosePath;
    double obervationProjectError;
    double filterProjectError;
    //点云过滤
    double xErrorThreshold;
    double yErrorThreshold;
    
};
#endif