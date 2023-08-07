#include "baseH.h"
#include "happly.h"
#include "Image.h"
#include "Point3d.h"
double getAngelOfTwoVector(std::vector<double>line1,std::vector<double>line2)
{
	double amb = line1[0] * line2[0] +line1[1] * line2[1] + line1[2] * line2[2];
	double sqa = sqrt(pow(line1[0], 2) + pow(line1[1], 2) + pow(line1[2], 2));
	double sqb = sqrt(pow(line2[0], 2) + pow(line2[1], 2) + pow(line2[2], 2));
	double cos_theta = amb / (sqa * sqb);
	return acos(cos_theta)*180;
}
double ImagePairAngle(cv::Mat Point3d,cv::Mat& opticsCenter1,cv::Mat& opticsCenter2){
    float sumAngle=0;
    for(int row=0;row<Point3d.rows;++row){
        std::vector<double>line1={Point3d.at<double>(row,0)-opticsCenter1.at<double>(0,0),
                                 Point3d.at<double>(row,1)-opticsCenter1.at<double>(1,0),
                                Point3d.at<double>(row,2)-opticsCenter1.at<double>(2,0)};
        std::vector<double>line2={Point3d.at<double>(row,0)-opticsCenter2.at<double>(0,0),
                                Point3d.at<double>(row,1)-opticsCenter2.at<double>(1,0),
                            Point3d.at<double>(row,2)-opticsCenter2.at<double>(2,0)};
        sumAngle+=getAngelOfTwoVector(line1,line2);
    }
    return sumAngle/Point3d.rows;
}
void createPath(std::string& path){
    if(access(path.c_str(), F_OK)!= 0){
        int ret=0;
        ret=mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(ret!=0)
            perror("mkdir:");
    }
}

void savePointCloud(std::vector<Point3dPtr>&reconstructPoint3ds,std::string &savePointCloudPath){
    // createPath(savePointCloudPath);
    std::vector<std::array<double,3>>Poisitions;
     std::vector<std::array<double,3>>Colors;
    for(auto Point3dPtr:reconstructPoint3ds){
        Poisitions.emplace_back(std::array<double,3>{ Point3dPtr->x_, Point3dPtr->y_, Point3dPtr->z_});
        Colors.emplace_back(Point3dPtr->Color);
    }
    happly::PLYData plyOut;
    plyOut.addVertexPositions(Poisitions);
    plyOut.addVertexColors(Colors);
    plyOut.write(savePointCloudPath, happly::DataFormat::Binary);
    return ;
}
void writePoint3d(std::vector<Point3dPtr>&Point3d,std::string& savePointPath){
    // createPath(savePointPath);
    std::ofstream of;
    of.open(savePointPath);
    for(auto iter:Point3d){
        of<<iter->x_<<" "<<iter->y_<<" "<<iter->z_;
        for(int i=0;i<iter->point2ds.size();++i){
            of<<" ";
            of<<iter->point2ds[i]->image_->imageIndex_;
        }
        of<<"\r\n";
    }
    of.close();
    return ;
}

void writePose(std::vector<ImagePtr>&Images,std::string& savePosePath){
    // createPath(savePosePath);
    std::ofstream of;
    of.open(savePosePath);
    for(int i=0;i<Images.size();++i){
        cv::Mat& R=Images[i]->R_;
        cv::Mat& T=Images[i]->T_;
        for(int row=0;row<3;++row){
            for(int col=0;col<3;++col){
                of<<R.at<double>(row,col)<<" ";
            }
        }
        for(int row=0;row<2;++row){
            of<<T.at<double>(row,0)<<" ";
        }
        of<<T.at<double>(2,0)<<"\r\n";
    }
    of.close();
    return ;
}


void saveTxt(std::vector<ImagePtr>&Images,std::vector<Point3dPtr>&Point3d,std::string& savePointPath,std::string& savePosePath){
    writePoint3d(Point3d,savePointPath);
    writePose(Images,savePosePath);
    return ;
}

bool HasPointPositiveDepth(cv::Mat& projectMatrix,cv::Mat& point3dH){
    double depth=projectMatrix.at<double>(2,0)*point3dH.at<double>(0,0)+
                 projectMatrix.at<double>(2,1)*point3dH.at<double>(1,0)+
                 projectMatrix.at<double>(2,2)*point3dH.at<double>(2,0)+
                 projectMatrix.at<double>(2,3)*point3dH.at<double>(3,0);
   

    return depth >=
         std::numeric_limits<double>::epsilon();
}
bool HasPointValidProject(Point2dPtr& point2d,cv::Mat& projectMatrix,cv::Mat& point3dH,double threshold){
    cv::Mat projetLoc=projectMatrix*point3dH;
    if(projetLoc.at<double>(2,0)==0)
        return false;
    return threshold*threshold>=
            (pow(projetLoc.at<double>(0,0)/projetLoc.at<double>(2,0)-point2d->x_,2)+
             pow(projetLoc.at<double>(1,0)/projetLoc.at<double>(2,0)-point2d->y_,2));
}
bool evalObservationDepthAndProject(Point2dPtr& point2d,double threshold){
    cv::Mat projectMatrix;
    ImagePtr& image=point2d->image_;
    Point3dPtr& point3dPtr=point2d->point3d_;
    cv::hconcat(image->R_,image->T_,projectMatrix);
    projectMatrix=image->Intrinsics_*projectMatrix;
    cv::Mat point3dH(4,1,CV_64FC1);
    point3dH.at<double>(0,0)=point3dPtr->x_;
    point3dH.at<double>(1,0)=point3dPtr->y_;
    point3dH.at<double>(2,0)=point3dPtr->z_;
    point3dH.at<double>(3,0)=1;
    return HasPointPositiveDepth(projectMatrix,point3dH) && HasPointValidProject(point2d,projectMatrix, point3dH,threshold);
    // return true;
}
bool isValildIntrisicMatrix(cv::Mat& Intrinsics,int rows,int cols){
    double cx=Intrinsics.at<double>(0,2);
    double cy=Intrinsics.at<double>(1,2);
    return cx<cols && cy<rows;
}
bool isValildRotatedMatrix(cv::Mat& R){
    for(int row=0;row<3;++row){
        for(int col=0;col<3;++col){
            if(std::abs(R.at<double>(row,col))>1)
                return false;
        }
    }
    return true;
}