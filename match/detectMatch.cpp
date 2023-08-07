#include "detectMatch.h"
#include <sys/types.h>
#include <sys/stat.h>

void FeatureExtractMatch::InitConfig(){
    SiftDector=cv::SIFT::create(numFeatures_);
    OrbDector=cv::ORB::create(numFeatures_);
    fbmatcher=cv::FlannBasedMatcher::create();
    return ;
}
void FeatureExtractMatch::featureExtractSIFT(ImagePtr&  image){
        SiftDector->detect(image->rgbImage_,image->featurePoints);
        SiftDector->compute(image->rgbImage_,image->featurePoints,image->featureDescribes);
    return ;
}
void FeatureExtractMatch::featureExtractORB(ImagePtr&  image){

        OrbDector->detect(image->rgbImage_,image->featurePoints);
        OrbDector->compute(image->rgbImage_,image->featurePoints,image->featureDescribes);
        //因为orb提取的特征的数据格式和flann不相符，所以需要转换
        image->featureDescribes.convertTo(image->featureDescribes,CV_32FC1);
    return ;
}
void FeatureExtractMatch::featureExtract(ImagePtr&  image){
    image->flag.lock();
    if(image->featurePoints.size()==0 || image->featureDescribes.data==nullptr){
        if( ExtractorName_=="SIFT"){
                featureExtractSIFT(image);
            }
        else if(ExtractorName_=="ORB"){
               featureExtractORB(image);
            }
     }
    image->flag.unlock();
}

void FeatureExtractMatch::featureMatch(ImagePtr&  image1,ImagePtr&  image2,std::vector<cv::DMatch>& goodMatch){
    featureExtract(image1);
      featureExtract(image2);
    std::vector<std::vector<cv::DMatch>>Match;
    fbmatcher->knnMatch(image1->featureDescribes,image2->featureDescribes,Match,flannK_);
    //用距离进行初步过滤
    for (size_t k = 0; k< Match.size();k++)
    {
    //只有当最接近匹配和次邻近匹配差距足够大，才说明此时的特征点是一个不错的特征点
    if (Match[k][0].distance < Match[k][1].distance*featureThreshold_){
        goodMatch.emplace_back(Match[k][0]);
      }
    }
    return ;
}

void FeatureExtractMatch::drawMatch(ImagePtr&  image1,ImagePtr&  image2,std::vector<cv::DMatch>&Match){
    std::string path="./output";
    if(access(path.c_str(), F_OK)!= 0){
        int ret=0;
        ret=mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(ret!=0)
            perror("mkdir:");
    }
    std::string savePath="./output/ransacDiffernt/";
    if(access(savePath.c_str(), F_OK)!= 0){
        int ret=0;
        ret=mkdir(savePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(ret!=0)
            perror("mkdir:");
    }
    cv::Mat imgMatches;
    cv::drawMatches(image1->rgbImage_, image1->featurePoints, image2->rgbImage_,image2->featurePoints,Match,imgMatches,cv:: Scalar::all(-1),
    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string savePictureName=savePath+std::to_string(image1->imageIndex_)+std::to_string(image2->imageIndex_)+".png";
    cv::imwrite(savePictureName,imgMatches);
    return ;

}
void FeatureExtractMatch::computeFMatrix(ImagePtr& image1,ImagePtr&  image2){
    std::vector<cv::DMatch> goodMatch;
    // std::cout<<"image1 :"<<image1->imageIndex_<<std::endl;
    // std::cout<<"image2 :"<<image2->imageIndex_<<std::endl;
    featureMatch(image1,image2,goodMatch);
     //得到匹配点的坐标
    std::vector<cv::Point2d> obj;
    std::vector<cv::Point2d> scene;
    for (size_t k = 0; k < goodMatch.size(); ++k)
      {
        obj.emplace_back(image1->featurePoints[goodMatch[k].queryIdx].pt);
        scene.emplace_back(image2->featurePoints[goodMatch[k].trainIdx].pt);
      }
    std::vector<uchar>inlinersOS,inlinersSO;
    image1->FMatrix[image2]=cv::findFundamentalMat(obj, scene, inlinersOS, cv::USAC_MAGSAC,3);
    image2->FMatrix[image1]=cv::findFundamentalMat(scene,obj, inlinersSO, cv::USAC_MAGSAC,3);
    //使用F矩阵使用对体几何约束再次进行过滤
    std::vector<cv::DMatch> ransacMatch;
    std::vector<cv::Point2d> filterobj;
    std::vector<cv::Point2d> filterscene;
    for(size_t m=0;m<goodMatch.size();++m){
        if(inlinersOS[m] && inlinersSO[m]){
            ransacMatch.emplace_back(goodMatch[m]);
            filterobj.emplace_back(image1->featurePoints[goodMatch[m].queryIdx].pt);
            filterscene.emplace_back(image2->featurePoints[goodMatch[m].trainIdx].pt);
        }
    }
    image1->pairPoints[image2]=std::move(filterobj);
    image2->pairPoints[image1]=std::move(filterscene);
   
    if(saveMatch_){
        drawMatch(image1,image2,ransacMatch);
    }
    return ;
}

void FeatureExtractMatch::work(){ 
    computeFMatrix(image1_,image2_);
    return ;
}