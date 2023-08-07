#ifndef POINT3D_H
#define POINT3D_H
#include "dataType.h"
#include "baseH.h"
#include "Point2d.h"
class Point2d;
class Image;
// struct HashImage;
// struct EqualImage;



// template<typename Point2dPtr> 
// struct HashPoint2d
// {
//     size_t operator()(const Point2dPtr P)const{
//         return std::hash<float>()(P->x_) ^ std::hash<float>()(P->y_);
//     }
// };
// template<typename Point2dPtr> 
// struct EqualPoint2d
// {
//     size_t operator()(const Point2dPtr P,const Point2dPtr Q)const{
//         return (P->x_==Q->x_)&&(P->y_==Q->y_);
//     }
// };

// template<typename ImagePtr> 
// struct HashImage
// {
//     size_t operator()(const ImagePtr& P)const{
//         return std::hash<int>()(P->imageIndex_);
//     }
// };
// template<typename ImagePtr> 
// struct EquaImage
// {
//     size_t operator()(const ImagePtr& P,const ImagePtr& Q)const{
//         return (P->imageIndex_==Q->imageIndex_);
//     }
// };


typedef std::shared_ptr<Point2d> Point2dPtr;
typedef std::shared_ptr<Image> ImagePtr;

class Point3d{
  public:
    Point3d(){}
    Point3d(PointType x,PointType y,PointType z):x_(x),y_(y),z_(z){}
  void addPoint2d(Point2dPtr& point2d);
  void addImages( ImagePtr& image);
  std::vector<ImagePtr>Images;
  std::vector<Point2dPtr>point2ds;
  double x_,y_,z_;
  std::array<double,3>Color;
  private:
  std::unordered_set<Point2dPtr,HashPoint2d<Point2dPtr>,EqualPoint2d<Point2dPtr>>point2dSet;
  std::unordered_set<ImagePtr,HashImage<ImagePtr>, EqualImage<ImagePtr>>ImageSet;
};

#endif