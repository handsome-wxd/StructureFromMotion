#include "sparseReconstruct.h"
#include "../base/Point3d.h"
#include "../base/util.h"
#include "../eval/evalProjection.h"
#include "../optm/CeresAdjustBundle.h"
void SparseReconstruct::addReconstructImage(ImagePtr& Image){
    if(reconstructImagesSet.find(Image)==reconstructImagesSet.end()){
        reconstructImagesSet.insert(Image);
        reconstructImages.emplace_back(Image);
    }
    return ;
}
void SparseReconstruct::addReconstructPoint3d(Point3dPtr& Point3d){
    if(reconstructPoint3dsSet.find( Point3d)==reconstructPoint3dsSet.end()){
        reconstructPoint3dsSet.insert( Point3d);
        reconstructPoint3ds.emplace_back(Point3d);
        // reconstructPoint3dPisition.emplace_back(std::array<double,3>{Point3d->x_,Point3d->y_,Point3d->z_});
    }
    return ;
}

void SparseReconstruct::PointTrack(ImagePtr& Image){
    //统计该图片的每个特征点所对应的图片个数即为该点的track
    for(auto iter:Image->pairPoints){
        for(auto point:iter.second){
            Image->TrackMap[point]+=1;
        }
    }
    return ;
}
void SparseReconstruct::computeTrack(std::vector<ImagePtr>&Images){
    for(auto ImagePtr:Images){
        PointTrack(ImagePtr);
    }
}
void SparseReconstruct:: InitTrackPointEdge(ImagePtr& image1,ImagePtr& image2){
    std::vector<cv::Point2d>&Image1FeaturePoint=image1->pairPoints[image2];
    std::vector<cv::Point2d>&Image2FeaturePoint=image2->pairPoints[image1];
    if(Image1FeaturePoint.size()!=Image2FeaturePoint.size()){
        std::cout<<"图像对特征点数不一致"<<std::endl;
        exit(-1);
    }
    for(int i=0;i<Image1FeaturePoint.size();++i){
        if(image1->TrackMap[Image1FeaturePoint[i]]>=trackTheshold_ &&
            image2->TrackMap[Image2FeaturePoint[i]]>=trackTheshold_){
                image1->PointsTrack[image2].emplace_back(Image1FeaturePoint[i]);
                image2->PointsTrack[image1].emplace_back(Image2FeaturePoint[i]);
            }
    }
    //printf("image1->PointsTrack[image2].size()=%d",image1->PointsTrack[image2].size());
    //如果Edge超过EdgeThreshold_将两个图像点连接在一起
    if(image1->PointsTrack[image2].size()>=edgeThreshold_){
        image1->Edges[image2]=true;
        image2->Edges[image1]=true;
    }
    else{
        image1->Edges[image2]=false;
        image2->Edges[image1]=false;
    }
    return ;
 }

void SparseReconstruct::createEdge(std::vector<ImagePtr>&Images){
    for(auto origImage:Images){
        for(auto iter:origImage->pairPoints){
            auto dstImage=iter.first;
            if(origImage->PointsTrack.find(dstImage)==origImage->PointsTrack.end()){
                InitTrackPointEdge(origImage,dstImage);
            }
        }
    }
}


void SparseReconstruct::ImagePairTrigulate(ImagePtr& image1,ImagePtr& image2){
        cv::Mat EPoint3d;
        cv::Mat& KMatrix1=image1->Intrinsics_;
        cv::Mat& KMatrix2=image2->Intrinsics_;
        std::vector<cv::Point2d>& Image1Point2d=image1->PointsTrack[image2];
        std::vector<cv::Point2d>& Image2Point2d=image2->PointsTrack[image1];
    
        //计算投影矩阵
        cv::Mat Image1Project,Image2Project;
        cv::hconcat(image1->R_,image1->T_,Image1Project);
        cv::hconcat(image2->R_, image2->T_,Image2Project);
        if(KMatrix1.type()!=Image1Project.type()){
            Image1Project.convertTo(Image1Project,KMatrix1.type());
        }
        if(KMatrix2.type()!=Image2Project.type()){
            Image2Project.convertTo(Image2Project,KMatrix2.type());
        }
        Image1Project=KMatrix1*Image1Project;
        Image2Project=KMatrix2*Image2Project;
        //计算三维点，齐次坐标
        cv::Mat HPoint3d;
        //进行三角化
        cv::triangulatePoints(Image1Project, Image2Project,Image1Point2d,Image2Point2d, HPoint3d);
        //将Point3d从齐次坐标转为欧式坐标
        cv::convertPointsFromHomogeneous(HPoint3d.t(),EPoint3d);
        for(int row=0;row<EPoint3d.rows;++row){
            Point2dPtr image1_point2dPtr=std::make_shared<Point2d>(Image1Point2d[row].x,Image1Point2d[row].y,image1);
            Point2dPtr image2_point2dPtr=std::make_shared<Point2d>(Image2Point2d[row].x,Image2Point2d[row].y,image2);
            Point3dPtr point3d;
            if(image1->Point2d_Point3d.count(image1_point2dPtr) && !image2->Point2d_Point3d.count(image1_point2dPtr)){
                point3d=image1->Point2d_Point3d[image1_point2dPtr];
                image2_point2dPtr->point3d_=point3d;
                if(!evalObservationDepthAndProject(image2_point2dPtr,obervationProjectError_))
                    continue;
                point3d->addPoint2d(image2_point2dPtr);
                image2->addPoint2d(image2_point2dPtr);
                image2->addPoint3d(point3d);
                point3d->addImages(image2);
                image2->Point2d_Point3d[image2_point2dPtr]=point3d;
            }
            else if(!image1->Point2d_Point3d.count(image1_point2dPtr) && image2->Point2d_Point3d.count(image1_point2dPtr)){
                point3d=image2->Point2d_Point3d[image2_point2dPtr];
                image1_point2dPtr->point3d_=point3d;
                if(!evalObservationDepthAndProject(image2_point2dPtr,obervationProjectError_))
                    continue;
                point3d->addPoint2d(image1_point2dPtr);
                image1->addPoint2d(image1_point2dPtr);
                image1->addPoint3d(point3d);
                point3d->addImages(image1);
                image1->Point2d_Point3d[image1_point2dPtr]=point3d;
            }
            else{
                point3d=std::make_shared<Point3d>(EPoint3d.at<double>(row,0),EPoint3d.at<double>(row,1),EPoint3d.at<double>(row,2));
                image1_point2dPtr->point3d_= point3d;
                image2_point2dPtr->point3d_= point3d;
                if(!evalObservationDepthAndProject(image1_point2dPtr,obervationProjectError_) || !evalObservationDepthAndProject(image2_point2dPtr,obervationProjectError_))
                    continue;
                point3d->addPoint2d(image2_point2dPtr);
                image2->addPoint2d(image2_point2dPtr);
                image2->addPoint3d(point3d);
                point3d->addImages(image2);
                image2->Point2d_Point3d[image2_point2dPtr]=point3d;

                point3d->addPoint2d(image1_point2dPtr);
                image1->addPoint2d(image1_point2dPtr);
                image1->addPoint3d(point3d);
                point3d->addImages(image1);
                image1->Point2d_Point3d[image1_point2dPtr]=point3d;

                 cv::Vec3b bgr = image1->rgbImage_.at<cv::Vec3b>( int(Image1Point2d[row].y),int(Image1Point2d[row].x));
                std::array<double,3>Color={double(bgr[2])/255,double(bgr[1])/255,double(bgr[0])/255};
                point3d->Color=std::move(Color);
                 addReconstructPoint3d(point3d);
            }  
        }
        image1->Edges[image2]=false;
        image2->Edges[image1]=false;
        addReconstructImage(image1);
        addReconstructImage(image2);
        return ;
}
/**
 * 选择一对图像，满足他们各自与其他图像总的trackPoint点和 图像对之间的trackPoint数量最大
*/
struct maxTrackImage
{
    bool operator()(std::vector<int>&a,std::vector<int>&b){
        return a[0]<b[0];
    }
};

void SparseReconstruct::selectImagePair(std::vector<ImagePtr>&Images){
    int num = Images.size();
    int maxPoint=0;
    std::unordered_map<ImagePtr,int,HashImage<ImagePtr>,EqualImage<ImagePtr>>sumTrackPoint;
    for (auto Image:Images)
    {
        for(auto iter:Image->PointsTrack){
            if(Image->Edges[iter.first]){
                sumTrackPoint[Image]+=iter.second.size();
            }
        }
    }
    std::priority_queue<std::vector<int>,std::vector<std::vector<int>>,maxTrackImage>minHeap;
    for(int i=0;i<num;++i){
        for(int j=i+1;j<num;++j){
            if(!Images[i]->Edges[Images[j]] || !Images[j]->Edges[Images[i]]){
                continue;
            }
            int compareValue=sumTrackPoint[Images[i]]*0.2+
                                sumTrackPoint[Images[j]]*0.2+
                                Images[i]->PointsTrack[Images[j]].size()*0.6;
            minHeap.push({compareValue,i,j});
            //选出值最大的n个图像对
            if(minHeap.size()>candiateImagePair_){
                minHeap.pop();
            }
        }
    }
    //进行筛选 图像对之间的夹角不能太大，也不能太小
    //通过计算三维点和相机光心之间的夹角进行计算
    while(!minHeap.empty()){
        std::vector<int>candidateImagePair=minHeap.top();
        minHeap.pop();
        //图像1 ，图像2
        ImagePtr image1=Images[candidateImagePair[1]];
        ImagePtr image2=Images[candidateImagePair[2]];
        cv::Mat& FMatrix=image1->FMatrix[image2];
        cv::Mat& KMatrix1=image1->Intrinsics_;
        cv::Mat& KMatrix2=image2->Intrinsics_;
        std::vector<cv::Point2d>& Image1Point2d=image1->PointsTrack[image2];
        std::vector<cv::Point2d>& Image2Point2d=image2->PointsTrack[image1];
      //计算本质矩阵
        // std::cout<<"kMatrix"<<KMatrix.type()<<std::endl;
        // std::cout<<"FMatrix"<<FMatrix<<std::endl;

        cv::Mat EMatrix=KMatrix2.t()*FMatrix*KMatrix1;
        //本质矩阵进行分解为R，t,是以Image1为世界坐标
        cv::Mat R,T;
        cv::recoverPose(EMatrix,Image1Point2d,Image2Point2d,KMatrix1,R,T);
        //计算投影矩阵
        cv::Mat Image1Project,Image2Project;
        cv::hconcat(cv::Mat::eye(3, 3, CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1),Image1Project);
        cv::hconcat(R, T,Image2Project);
        Image1Project=KMatrix1*Image1Project;
        Image2Project=KMatrix2*Image2Project;
        //计算三维点，齐次坐标
        cv::Mat HPoint3d;
        //进行三角化
        cv::triangulatePoints(Image1Project, Image2Project,Image1Point2d,Image2Point2d, HPoint3d);
        //将Point3d从齐次坐标转为欧式坐标
        cv::Mat EPoint3d;
        cv::convertPointsFromHomogeneous(HPoint3d.t(),EPoint3d);

        //计算两个视图之间的夹角
        //第一个视图的光心位置
        cv::Mat worldCenter=cv::Mat::zeros(3, 1, CV_64FC1);
        //第二个视图的光心位置
        cv::Mat photoCenter=-R.inv()*T;
        double angle= ImagePairAngle(EPoint3d,worldCenter,photoCenter);
        if(angle>minAngleTheshold_ && angle<maxAngleTheshold_){
            image1->R_=cv::Mat::eye(3, 3, CV_64FC1);
            image1->T_=cv::Mat::zeros(3, 1, CV_64FC1);
            image2->R_=R;
            image2->T_=T;
            
            ImagePairTrigulate(image1,image2);
            return ;
        }
     }
      std::cout<<"在选择初始化图像对时,没有符合条件的图像对"<<std::endl;
      exit(-1);
    }

void SparseReconstruct::findNextEdge(std::vector<ImagePtr>&Images,ImagePtr& alreadyImage,ImagePtr& nextImage){
    int maxPointNum=0;
    std::unordered_map<cv::Point2d,Point3dPtr,HashCVPoint2d>final_point2d_point3dPtr;
    for(auto imagePtr:Images){
        //跳过已经重建的图像
        if(reconstructImagesSet.find(imagePtr)!=reconstructImagesSet.end()){
            continue;
        }
        std::unordered_map<cv::Point2d,Point3dPtr,HashCVPoint2d>point2d_point3dPtr;
        int maxNum=0;
        for(auto reconstructImage:reconstructImages){
            //如果两个图像之间没有Edge，则跳出
            if(!imagePtr->Edges[reconstructImage] || !reconstructImage->Edges[imagePtr]){
                continue;
            }
            int temp=0;
            std::vector<cv::Point2d>& trackPoint=reconstructImage->PointsTrack[imagePtr];
            for(int i=0;i<trackPoint.size();++i){
                Point2dPtr tempPoint2dPtr=std::make_shared<Point2d>(trackPoint[i].x,trackPoint[i].y);
                 if(reconstructImage->isReconstructPoint2d(tempPoint2dPtr)){
                    Point3dPtr tempPoint3dPtr=reconstructImage->Point2d_Point3d[tempPoint2dPtr];
                    cv::Point2d tempPoint2d=imagePtr->PointsTrack[reconstructImage][i];
                    point2d_point3dPtr[tempPoint2d]=tempPoint3dPtr;
                    temp+=1;
                  }
                }
                if(maxNum<temp){
                    maxNum=temp;
                    alreadyImage=reconstructImage;
                }
             }
               //获得交集最大的图像，和对应的三维点和二维点
              if(maxPointNum<point2d_point3dPtr.size()){
                    maxPointNum=point2d_point3dPtr.size();
                    nextImage=imagePtr;
                    final_point2d_point3dPtr=point2d_point3dPtr;
              }
    }

    if(maxPointNum>3){
        std::vector<cv::Point3d>point3ds(maxPointNum);
        std::vector<cv::Point2d>point2ds(maxPointNum);
        int index=0;
        for(auto iter:final_point2d_point3dPtr){
            point2ds[index]=iter.first;
            point3ds[index]=cv::Point3d(iter.second->x_,iter.second->y_,iter.second->z_);
            index+=1;
        }
       //PnP求解nextImage位姿
        cv::solvePnP(point3ds,point2ds,nextImage->Intrinsics_,cv::Mat(),nextImage->R_,nextImage->T_,false,0);
        cv::Rodrigues(nextImage->R_,nextImage->R_);
        // //将二维点和三维点进行绑定
        index=0;
         for(auto iter:final_point2d_point3dPtr){
            Point2dPtr tempPoint2d=std::make_shared<Point2d>(iter.first.x,iter.first.y,nextImage,iter.second);
            if(!evalObservationDepthAndProject(tempPoint2d,obervationProjectError_)){
                continue;
            }
            iter.second->addPoint2d(tempPoint2d);
            iter.second->addImages(nextImage);
            nextImage->addPoint2d(tempPoint2d);
            nextImage->addPoint3d(iter.second);
            nextImage->Point2d_Point3d[tempPoint2d]=iter.second;
            //point3d.emplace_back(finalPoint3dPtr[i]->x_,finalPoint3dPtr[i]->y_,finalPoint3dPtr[i]->z_);
        }
    }
    else{
        alreadyImage=nullptr;
        nextImage=nullptr;
    }
    return ;
}
//滤除无效图像
void SparseReconstruct::filterImage(){
    if(reconstructImages.size()<20)
        return ;
    std::vector<ImagePtr>filterImages;
    reconstructImagesSet.clear();
    for(int i=0;i<reconstructImages.size();++i){
        if(reconstructImages[i]->Point3ds.size()==0)
            continue;
        if(!isValildIntrisicMatrix(reconstructImages[i]->Intrinsics_,reconstructImages[i]->rows,reconstructImages[i]->cols))
            continue;
        if(!isValildRotatedMatrix(reconstructImages[i]->R_))
            continue;
        filterImages.emplace_back(reconstructImages[i]);
        reconstructImagesSet.insert(reconstructImages[i]);
    }
    reconstructImages.clear();
    reconstructImages=filterImages;
    return ;
}

//滤除无效点
void SparseReconstruct::filterPoint(){
    std::vector<Point3dPtr>filterPoint3d;
    reconstructPoint3dsSet.clear();
    for(int i=0;i<reconstructPoint3ds.size();++i){
        bool flag=true;
        for(auto point2d:reconstructPoint3ds[i]->point2ds){
           if(!evalObservationDepthAndProject(point2d, filterProjectError_)){
            flag=false;
            break;
           }
        }
        if(flag){
            filterPoint3d.emplace_back(reconstructPoint3ds[i]);
            reconstructPoint3dsSet.insert(reconstructPoint3ds[i]);
        }
    }
    reconstructPoint3ds.clear();
    reconstructPoint3ds=filterPoint3d;
    return ;
}

void SparseReconstruct::multiImageEstimatePose(std::vector<ImagePtr>&Images, PointCloundOptmPtr& pointCloundOptm){
    //计算符合track阈值的点,并连接符合Edege阈值的两个图像
    // printf("ok");
    computeTrack(Images);
    createEdge(Images);
   
    //挑选出一对符合条件的初始重建图像对
    selectImagePair(Images);
    SolveCeresBA(reconstructImages,reconstructPoint3ds);
   
    filterPoint();
    filterImage();
    while(true){
        ImagePtr nextImage=nullptr;
        ImagePtr alreadyImage=nullptr;
        findNextEdge(Images,alreadyImage,nextImage);
        if(alreadyImage==nullptr || nextImage==nullptr){
            break;
        }
        ImagePairTrigulate(alreadyImage,nextImage);
        //ajustment boundle
        SolveCeresBA(reconstructImages,reconstructPoint3ds);
        // filterPoint();
        // filterImage();
    }
    evalProjection eval(reconstructPoint3ds);
    eval.errorReport();
    // std::vector<Point3dPtr>filterPoint3ds;
    // pointCloundOptm->filterErrorPoint(reconstructPoint3ds,filterPoint3ds);
    savePointCloud(reconstructPoint3ds,savePointCloudPath_);
    //保存三维点信息为txt
    saveTxt(reconstructImages,reconstructPoint3ds,savePointPath_,savePosePath_);
}

