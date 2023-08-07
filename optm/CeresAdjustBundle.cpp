#include "CeresAdjustBundle.h"



void poseToVec(std::vector<ImagePtr>&Images,double *ptr, int cameraBlockSize){
    int num=Images.size();
    for(int i=0;i<num;++i){
        // int a=Images[i]->R_.type();
        // int b=Images[i]->T_.type();
        // std::cout<<"R"<<a<<std::endl;
        // std::cout<<"T"<<b<<std::endl;
        // std::cout<<Images[i]->R_<<std::endl;
        // std::cout<<Images[i]->T_<<std::endl;
        cv::Mat r;
        int index=0;
        cv::Rodrigues(Images[i]->R_,r);
        // std::cout<<r<<std::endl;
        //将R赋值到vec中0-2
        for(int j=0;j<3;++j){
            ptr[i*cameraBlockSize+index]=r.at<double>(j,0);
            index+=1;
            
        }
        //将T赋值到vec中3-5
        for(int j=0;j<3;++j){
            ptr[i*cameraBlockSize+index]=Images[i]->T_.at<double>(j,0);
            index+=1;
        }
        // //将K赋值给vec中的6-15
        // for(int row=0;row<3;++row){
        //     for(int col=0;col<3;++col){
        //           ptr[i*cameraBlockSize+index]=Images[i]->Intrinsics_.at<double>(row,col);
        //           index+=1;
        //     }
        // }
    }
}

//向量转相机位姿
void vecToPose(double *ptr,std::vector<ImagePtr>&Images, int cameraBlockSize){
    int num=Images.size();
    for(int i=0;i<num;++i){
        //更新R
        cv::Mat r(3,1,CV_64FC1);
        int index=0;
        for(int j=0;j<3;++j){
            r.at<double>(j,0)=ptr[i*cameraBlockSize+index];
            index+=1;
        }
        cv::Rodrigues(r,Images[i]->R_);
         //更新T
        for(int j=0;j<3;++j){
           Images[i]->T_.at<double>(j,0)=ptr[i*cameraBlockSize+index];
           index+=1;
        }
        //更新
      
    }
}


//将点转化为一维向量
void point3dToVec(std::vector<Point3dPtr>&Point3ds,double *pointPtr, int pointBlockSize ){
    int num=Point3ds.size();
    for(int i=0;i<num;++i){
        pointPtr[i*pointBlockSize+0]=Point3ds[i]->x_;
        pointPtr[i*pointBlockSize+1]=Point3ds[i]->y_;
        pointPtr[i*pointBlockSize+2]=Point3ds[i]->z_;
    }
}
//一维向量转3d点

void vecToPoint3d(double *pointPtr,std::vector<Point3dPtr>&Point3ds, int pointBlockSize ){
     int num=Point3ds.size();
    for(int i=0;i<num;++i){
        Point3ds[i]->x_=pointPtr[i*pointBlockSize+0];
        Point3ds[i]->y_=pointPtr[i*pointBlockSize+1];
        Point3ds[i]->z_=pointPtr[i*pointBlockSize+2];
    }
}

/***
 * 输入相机位姿，3d点的坐标
*/
void SolveCeresBA(std::vector<ImagePtr>& Images,std::vector<Point3dPtr>& Point3ds){
    double f1=Images[0]->Intrinsics_.at<double>(0,0),f2=Images[0]->Intrinsics_.at<double>(1,1),
    cx=Images[0]->Intrinsics_.at<double>(0,2),cy=Images[0]->Intrinsics_.at<double>(1,2);
     //三维点的个数
    const int pointNum=Point3ds.size();
    //相机位姿的个数
    const int cameraNum=Images.size();
    int cameraBlockSize=17;//R 3 T 3  3+3+9+2=17
    int pointBlockSize=3;//x y z 3
    std::unordered_map<int,int>imageIndexMap;
    for(int i=0;i<Images.size();++i){
        imageIndexMap[Images[i]->imageIndex_]=i;
    }
    //将相机位姿转为向量，将三维点转为向量
    double posePtr[cameraBlockSize*cameraNum];
    double pointPtr[pointBlockSize*pointNum];
    poseToVec(Images,posePtr,cameraBlockSize);
    point3dToVec(Point3ds,pointPtr,pointBlockSize);
    std::cout<<std::endl;
    ceres::Problem problem;
    int observeation=0;

    for(int i=0;i<pointNum;++i){
        for(auto point2d:Point3ds[i]->point2ds){
            observeation+=1;
            ceres::CostFunction *cost_function;
            cost_function = SnavelyReprojectionError::Create(point2d->x_, point2d->y_,f1,f2,cx,cy);
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            double *camera = posePtr + cameraBlockSize * imageIndexMap[point2d->image_->imageIndex_];
            double *point =pointPtr + pointBlockSize * i;
            problem.AddResidualBlock(cost_function, loss_function, camera, point);
        }
    }
    //show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << cameraNum << " cameras and "
              << pointNum << " points. " << std::endl;
    std::cout << "Forming " << observeation << " observations. " << std::endl;
    std::cout << "Solving ceres BA ... " << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout<<"更新相机参数和三维点坐标"<<std::endl;
    vecToPose(posePtr,Images,cameraBlockSize);
    vecToPoint3d(pointPtr,Point3ds,pointBlockSize);
   
    return ;
}
