#include "reconstruct.h"
#include "../base/util.h"

#include "../denseReconstruct/DenseMatch.h"
void Reconstruct::InitConfig(){
    imagesInfo=std::make_shared< InputImageInformaton>(config_->ImagesPath,config_->IntrinsicsPath,config_->isEvalIntrisic);
    sparseReconstruct=std::make_shared<SparseReconstruct>(config_->trackTheshold,config_->edgeThreshold,config_->candiateImagePair,
    config_->minAngleTheshold,config_->maxAngleTheshold,config_->obervationProjectError,config_->filterProjectError,
    config_->savePointCloudPath,config_->savePointPath,config_->savePosePath);
    pointCloundOptm=std::make_shared<PointCloundOptm>(config_->xErrorThreshold,config_->yErrorThreshold);
   // FeatureExtractThreadpool=new Threadpool<FeatureExtractMatchPtr>(threadNum,config_->matchRequestNum);
    //threadpool=new Threadpool<FeatureExtractMatchPtr>(threadNum,config_->matchRequestNum);
    return ;
}
void Reconstruct::InitImagesInfo(){
    imagesInfo->LoadImageInfo();
    return ;
}
void Reconstruct::ExtractMatch(){
    int imgNum=imagesInfo->Images.size();
    int threadNum=config_->matchThreadNum<imgNum*imgNum/2? config_->matchThreadNum:imgNum*imgNum/2;
    FeatureExtractThreadpool=new Threadpool<FeatureExtractMatchPtr>(threadNum,config_->matchRequestNum);
    for(int i=0;i<imgNum;++i){
        for(int j=i+1;j<imgNum;++j){
              FeatureExtractMatchPtr featureExtractMatch=std::make_shared<FeatureExtractMatch>(config_->ExtractorName,
                                        config_->numFeatures,imagesInfo->Images[i],imagesInfo->Images[j],config_->saveMatch,
                                        config_->featureThreshold,config_->flannK);
              featureExtractMatch->InitConfig();
              FeatureExtractThreadpool->append(featureExtractMatch);
            //   sleep(0.3);
        }
    }
    //等到匹配结束
    // while(!threadpool->workEmpty()){
    //     std::cout<<"";
    // }
    FeatureExtractThreadpool->waitAllTasks();
    // threadpool->waitAllTasks();
    // delete threadpool;
    // sleep(0.3);
    return ;
}

void Reconstruct::SpairReconstruct(){
    //多视图位姿估计
    sparseReconstruct->multiImageEstimatePose(imagesInfo->Images,pointCloundOptm);
    return ;
}

void Reconstruct::DenseReconstruct(){
    DenseMatch(sparseReconstruct->reconstructImages);
    return ;
}


