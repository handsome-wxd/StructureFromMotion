#include "Image.h"
#include "Point2d.h"
#include "Point3d.h"
// void Image::setIntrinsics(std::string& kPath){
//     Intrinsics_.create(3,3,CV_32FC1);
//     readK(kPath,Intrinsics_);
//     return ;
// }
// bool Image::isExtract(){
//     flag.wait();
//     if(featureDescribes.data==nullptr){

//     }
//     flag.post()    
// }
void Image::setIntrinsics(cv::Mat& Intrinsics){
    Intrinsics_=Intrinsics;
    return ;
}
void Image::setR(cv::Mat& R){
    R_=R;
}
void Image::setT(cv::Mat& T){
    T_=T;
}
bool Image::isReconstructPoint2d(Point2dPtr point2d){
    return Point2dSet.find(point2d)!=Point2dSet.end();
}
bool Image::isReconstructPoint3d(Point3dPtr point3d){
    return Point3dSet.find(point3d)!=Point3dSet.end();
}

void Image::addPoint3d(Point3dPtr point3d){
    if(!isReconstructPoint3d(point3d)){
        Point3dSet.insert(point3d);
        Point3ds.emplace_back(point3d);
    }
}
void Image::addPoint2d(Point2dPtr point2d){
     if(!isReconstructPoint2d(point2d)){
        Point2dSet.insert(point2d);
        Point2ds.emplace_back(point2d);
    }
}

void Image::setProjectMatrix(){
    if(R_.data==nullptr){
        std::cout<<"未初始化旋转变量"<<std::endl;
    }
    else if(T_.data==nullptr){
         std::cout<<"未初始化平移变量"<<std::endl;
    }
    ProjectMatrix.create(3,4,CV_32FC1);
    cv::hconcat(R_,T_,ProjectMatrix);
    return ;
}