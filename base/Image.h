#ifndef IMAGE_H
#define IMAGE_H
#include <opencv2/opencv.hpp>
#include "baseH.h"
#include "dataType.h"
#include "../threadPool/threadpool.h"
#define OPENCV_TRAITS_ENABLE_DEPRECATED


class Point2d;
class Point3d;
class Image;


template<typename Point3dPtr> 
struct HashPoint3d
{
    size_t operator()(const Point3dPtr P)const{
        return std::hash<float>()(P->x_) ^ std::hash<float>()(P->y_)^std::hash<float>()(P->z_);
    }
};
template<typename  Point3dPtr> 
struct EqualPoint3d
{
    size_t operator()(const Point3dPtr& P,const Point3dPtr& Q)const{
        return (P->x_==Q->x_)&&(P->y_==Q->y_)&&(P->z_==Q->z_);
    }
};

template<typename Point2dPtr> 
struct HashPoint2d
{
    size_t operator()(const Point2dPtr P)const{
        return std::hash<float>()(P->x_) ^ std::hash<float>()(P->y_);
    }
};
template<typename Point2dPtr> 
struct EqualPoint2d
{
    size_t operator()(const Point2dPtr P,const Point2dPtr Q)const{
        return (P->x_==Q->x_)&&(P->y_==Q->y_);
    }
};
template<typename ImagePtr> 
struct HashImage
{
    size_t operator()(const ImagePtr& P)const{
        return std::hash<int>()(P->imageIndex_);
    }
};
template<typename ImagePtr> 
struct EqualImage
{
    size_t operator()(const ImagePtr& P,const ImagePtr& Q)const{
        return (P->imageIndex_==Q->imageIndex_);
    }
};

typedef std::shared_ptr<Point2d> Point2dPtr;
typedef std::shared_ptr<Point3d> Point3dPtr;
typedef std::shared_ptr<Image> ImagePtr;


class Image{
public:
    Image():imageName_(""),imageIndex_(-1),rows(0),cols(0){}
    Image(ImageNameType imageName,ImageIndexType imageIndex,cv::Mat &rgbImage,cv::Mat &grayImage):imageName_(imageName),imageIndex_(imageIndex),
    rgbImage_(rgbImage),grayImage_(grayImage){
        rows=rgbImage_.rows;
        cols=rgbImage_.cols;
    }
    void setIntrinsics(std::string& kPath);
    void setIntrinsics(cv::Mat& Intrinsics);
    void setR(cv::Mat& R);
    void setT(cv::Mat& T);
    bool isReconstructPoint2d(Point2dPtr point2d);
    bool isReconstructPoint3d(Point3dPtr point3d);
    void addPoint3d(Point3dPtr point3d);
    void addPoint2d(Point2dPtr point2d);
    void setProjectMatrix();
    bool isExtract();
     bool operator==(const Image& other)const{//重载operator==(),若没有重载==则定义 unordered_map 时需要isEqual
      return imageIndex_==other.imageIndex_;
     }
    ImageNameType imageName_;
    ImageIndexType imageIndex_;
    cv::Mat Intrinsics_;
    cv::Mat R_;
    cv::Mat T_;
    cv::Mat distCoeffs_;
    cv::Mat ProjectMatrix;
    int rows;
    int cols;
    cv::Mat rgbImage_;
    cv::Mat grayImage_;
    cv::Mat DepthMap;
    std::vector<Point3dPtr>Point3ds;
    std::vector<Point2dPtr>Point2ds;
    
    //存储本图片和Image 对应的特征点
    std::unordered_map<ImagePtr,std::vector<cv::Point2d>,HashImage<ImagePtr>,EqualImage<ImagePtr>>pairPoints;
    std::unordered_map<ImagePtr,cv::Mat,HashImage<ImagePtr>,EqualImage<ImagePtr>>FMatrix;
    //特征描述符 特征描述子
    std::vector<cv::KeyPoint>featurePoints;
    cv::Mat featureDescribes;
   
    //存储满足TrackThreshold的点
    std::unordered_map<cv::Point2d,int,HashCVPoint2d>TrackMap;
    std::unordered_map<ImagePtr,std::vector<cv::Point2d>,HashImage<ImagePtr>,EqualImage<ImagePtr>>PointsTrack;
    std::unordered_map<ImagePtr,std::vector<bool>,HashImage<ImagePtr>,EqualImage<ImagePtr>>PointsTrackFlag;
    std::unordered_map<Point2dPtr,Point3dPtr,HashPoint2d<Point2dPtr>,EqualPoint2d<Point2dPtr>>Point2d_Point3d;
    std::unordered_map<ImagePtr,bool,HashImage<ImagePtr>,EqualImage<ImagePtr>>Edges;
    locker flag;
private:
    std::unordered_set<Point3dPtr,HashPoint3d<Point3dPtr>,EqualPoint3d<Point3dPtr>>Point3dSet;
    std::unordered_set<Point2dPtr,HashPoint2d<Point2dPtr>,EqualPoint2d<Point2dPtr>>Point2dSet;
  

};



#endif