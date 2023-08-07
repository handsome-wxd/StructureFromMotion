#include "Point3d.h"
#include "Point2d.h"
#include "Image.h"

void Point3d::addImages(ImagePtr& image){
    if(ImageSet.find(image)==ImageSet.end()){
        ImageSet.insert(image);
        Images.emplace_back(image);
    }
        return ;
}
 
void Point3d::addPoint2d(Point2dPtr & point2d){
    if(point2dSet.find(point2d)==point2dSet.end()){
        point2ds.emplace_back(point2d);
        point2dSet.insert(point2d);
    }
    return ;
  }