#ifndef DATA_H
#define DATA_H
#include "../base/Image.h"

class InputImageInformaton{
public:
    InputImageInformaton(std::string& ImagesPath,std::string IntrinsicaPath,bool isEvalIntrisic=false):
        ImagesPath_(ImagesPath),IntrinsicsPath_(IntrinsicaPath),isEvalIntrisic_(isEvalIntrisic){}
    
    void LoadImageInfo();
    std::vector<ImagePtr>Images;
    private:
    void evalEvalIntrisic(const std::string& filePath,cv::Mat& Intrinsic);
    void ImageLoad();
    void readK(const std::string& filePath,cv::Mat &Matrix);
    std::string ImagesPath_;
    std::string IntrinsicsPath_;
    std::string evalIntrinsicaPath_;
   
    bool isEvalIntrisic_;

};


#endif