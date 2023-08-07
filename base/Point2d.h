#ifndef POINT2D_H
#define POINT2D_H

#include "dataType.h"
#include "baseH.h"
#include "Point3d.h"
#include "Image.h"

class Point3d;
class Image;
class Point2d{
    public:
        Point2d(double x,double y):x_(x),y_(y){}
        Point2d(double x,double y,ImagePtr& image):x_(x),y_(y),image_(image){}
        Point2d(double x,double y,ImagePtr& image,Point3dPtr& point3d):x_(x),y_(y),image_(image),point3d_(point3d){}
        ImagePtr image_;
        Point3dPtr point3d_;
        double x_,y_;
};


#endif