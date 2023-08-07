#include "evalProjection.h"
#include "../base/Point3d.h"
#include <numeric>

void evalProjection::computeError(){
    for(Point3dPtr point3dptr:point3ds_){
        cv::Mat point3dMat(4,1,CV_64FC1);
        point3dMat.at<double>(0,0)=point3dptr->x_;
        point3dMat.at<double>(1,0)=point3dptr->y_;
        point3dMat.at<double>(2,0)=point3dptr->z_;
        point3dMat.at<double>(3,0)=1;
        for(Point2dPtr point2dPtr:point3dptr->point2ds){
          cv::Mat M;
          cv::hconcat(point2dPtr->image_->R_,point2dPtr->image_->T_,M);
          cv::Mat projectPoint=point2dPtr->image_->Intrinsics_*M*point3dMat;
          xError.emplace_back(std::abs(point2dPtr->x_-projectPoint.at<double>(0,0)/projectPoint.at<double>(2,0)));
          yError.emplace_back(std::abs(point2dPtr->y_-projectPoint.at<double>(1,0)/projectPoint.at<double>(2,0)));
        }
    }
    return ;
}


void evalProjection::errorStatistics(){
     if(xError.size()==0 || yError.size()==0){
        return ;
     }
     for(auto& i : xError)
        i = abs(i);
    for(auto& i : yError)
        i = abs(i);
     // mean value
    double mean_x = std::accumulate(begin(xError), end(xError), 0.0) / xError.size();
    double mean_y = std::accumulate(begin(yError), end(yError), 0.0) / yError.size();
    
    double min_x = *std::min_element(xError.begin(),xError.end());
    double max_x = *std::max_element(xError.begin(),xError.end());
    double min_y = *std::min_element(yError.begin(),yError.end());
    double max_y = *std::max_element(yError.begin(),yError.end());
    
    // double cost_hori = 0.0;
    // for(auto i= 0;i!=errs_x.size();i++){
    //     cost_hori += errs_x[i] * errs_x[i] + errs_y[i] * errs_y[i];
    // }
    // double mean_cost_horizontal = sqrt(cost_hori/(errs_x.size()));
    double cost_sum_x = 0.0;
    double cost_sum_y = 0.0;
    for(auto i= 0;i!=xError.size();i++){
        cost_sum_x += xError[i] * xError[i];
        cost_sum_y += yError[i] * yError[i];
    }
    double RMSE_x = sqrt( cost_sum_x/xError.size());
    double RMSE_y = sqrt( cost_sum_y/yError.size());
                                     
    double RMSE_horizontal = sqrt( cost_sum_x/xError.size() +
                                     cost_sum_y/yError.size());


    std::cout << std::right << std::setw(16) << "MAE in x direction: ";
    std::cout << std::right << std::setprecision(6)
            << mean_x
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "MAE  in y direction: ";
    std::cout << std::right << std::setprecision(6)
            << mean_y
            << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "min abs error in x direction: ";
    std::cout << std::right << std::setprecision(6)
            << min_x
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "max abs error in x direction: ";
    std::cout << std::right << std::setprecision(6)
            << max_x
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "min abs error in y direction: ";
    std::cout << std::right << std::setprecision(6)
            << min_y
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "max abs error in y direction: ";
    std::cout << std::right << std::setprecision(6)
            << max_y
            << " [px]" << std::endl;


    std::cout << std::right << std::setw(16) << "RMSE in the x direction: ";
    std::cout << std::right << std::setprecision(6)
            << RMSE_x
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "RMSE in the y direction: ";
    std::cout << std::right << std::setprecision(6)
            << RMSE_y
            << " [px]" << std::endl;
    std::cout << std::right << std::setw(16) << "RMSE  in the horizontal direction: ";
    std::cout << std::right << std::setprecision(6)
            << RMSE_horizontal
            << " [px]" << std::endl;
}


void evalProjection::errorReport(){
        computeError();
        errorStatistics();
        return ;
}