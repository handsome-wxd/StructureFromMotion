#include "PointCloudOptm.h"
#include "../base/Point3d.h"
void PointCloundOptm::filterErrorPoint(std::vector<Point3dPtr>& point3ds,std::vector<Point3dPtr>& filterPoint3ds){
     for(Point3dPtr point3dptr:point3ds){
        cv::Mat point3dMat(4,1,CV_64FC1);
        point3dMat.at<double>(0,0)=point3dptr->x_;
        point3dMat.at<double>(1,0)=point3dptr->y_;
        point3dMat.at<double>(2,0)=point3dptr->z_;
        point3dMat.at<double>(3,0)=1;
        double xError=0;
        double yError=0;
        int num=0;
        for(Point2dPtr point2dPtr:point3dptr->point2ds){
          cv::Mat M;
          cv::hconcat(point2dPtr->image_->R_,point2dPtr->image_->T_,M);
          cv::Mat projectPoint=point2dPtr->image_->Intrinsics_*M*point3dMat;
          xError+=std::abs(point2dPtr->x_-projectPoint.at<double>(0,0)/projectPoint.at<double>(2,0));
          yError+=std::abs(point2dPtr->x_-projectPoint.at<double>(1,0)/projectPoint.at<double>(2,0));
          num+=1;
        }
        if(xError/num>xErrorThreshold_ || yError/num>yErrorThreshold_){
            continue;
        }
        filterPoint3ds.emplace_back(point3dptr);
    }
    return ;
}