#ifndef FRONT_H
#define FRONT_H
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "../base/baseH.h"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/xfeatures2d.hpp>
#include "../base/Image.h"
#include "ImgInfo.h"
#include "../threadPool/locker.h"
typedef std::shared_ptr<InputImageInformaton> InputImageInformatonPtr;
class FeatureExtractMatch{
public:
    FeatureExtractMatch():ExtractorName_("SIFT"),numFeatures_(5000),featureThreshold_(1),flannK_(2),saveMatch_(false){}
    FeatureExtractMatch(std::string& ExtractorName,int numFeatures,ImagePtr& image1,ImagePtr& image2,bool saveMatch=false,float featureThreshold=1,int flannK=2):
    numFeatures_(numFeatures),ExtractorName_(ExtractorName),featureThreshold_(featureThreshold),flannK_(flannK),
    saveMatch_(saveMatch),image1_(image1),image2_(image2){}

    void work();
    void InitConfig();
    void featureExtract(ImagePtr&  image);
private:
   
    void featureMatch(ImagePtr& image1,ImagePtr& image2,std::vector<cv::DMatch>& goodMatch);
    void computeFMatrix(ImagePtr& image1,ImagePtr& image2);
    void drawMatch(ImagePtr& image1,ImagePtr& image2,std::vector<cv::DMatch>&Match);
    void featureExtractSIFT(ImagePtr& image);
    void featureExtractORB(ImagePtr& image1);
    // std::shared_ptr<InputImageInformaton>imgInfo_;
    int numFeatures_;
    std::string ExtractorName_;
    cv::Ptr<cv::SIFT> SiftDector;
    cv::Ptr<cv::ORB> OrbDector;
    cv::Ptr<cv::FlannBasedMatcher> fbmatcher;
    float featureThreshold_;
    int flannK_;
    bool saveMatch_;
    ImagePtr image1_;
    ImagePtr image2_;
    

};
#endif