#ifndef RECONSTRUCT_H
#define RECONSTURCT_H
#include "../match/ImgInfo.h"
#include "../match/detectMatch.h"
#include "../base/Config.h"
#include "../optm/PointCloudOptm.h"
#include "sparseReconstruct.h"
#include "../threadPool/threadpool.h"
typedef std::shared_ptr< FeatureExtractMatch>  FeatureExtractMatchPtr;
typedef std::shared_ptr< SparseReconstruct>  SparseReconstructPtr;
typedef std::shared_ptr<Config>ConfigPtr;

class Reconstruct{
public:
    Reconstruct(ConfigPtr config):config_(config){}
    void InitConfig();
    void InitImagesInfo();
    void ExtractMatch();
    void SpairReconstruct();
    void DenseReconstruct();
    ~Reconstruct(){
        delete FeatureExtractThreadpool;
    }
private:
    ConfigPtr config_;
    InputImageInformatonPtr imagesInfo;
    Threadpool<FeatureExtractMatchPtr>* FeatureExtractThreadpool=nullptr;
   // Threadpool<FeatureExtractMatchPtr>* DenseMatchThreadpool=nullptr;

    SparseReconstructPtr  sparseReconstruct;
    PointCloundOptmPtr pointCloundOptm;
};
#endif