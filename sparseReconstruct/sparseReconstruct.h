#ifndef SPARSERECONSTURCT_H
#define SPARSERECONSTURCT_H
//定义稀疏重建类
#include "../base/Image.h"
#include "../optm/PointCloudOptm.h"
typedef std::shared_ptr<PointCloundOptm>PointCloundOptmPtr;

class SparseReconstruct{
public:
   SparseReconstruct(int trackTheshold=3,int edgeThreshold=100,int candiateImagePair=5,double minAnlgeThreshold=5,double maxAngleThreshold=60, double obervationProjectError=10,
   double filterProjectError=4.0,std::string savePointCloudPath="./outPut/sparse.ply",std::string savePointPath="./outPut/point.txt",std::string savePosePath="./outPut/pose.txt")
   :trackTheshold_(trackTheshold),edgeThreshold_(edgeThreshold),candiateImagePair_(candiateImagePair),
   minAngleTheshold_(minAnlgeThreshold),maxAngleTheshold_(maxAngleThreshold),obervationProjectError_(obervationProjectError),
   filterProjectError_(filterProjectError),savePointCloudPath_(savePointCloudPath),savePointPath_(savePointPath),savePosePath_(savePosePath){}

   void multiImageEstimatePose(std::vector<ImagePtr>&Images, PointCloundOptmPtr& pointCloundOptm);
   std::vector<ImagePtr>reconstructImages;
   std::vector<Point3dPtr>reconstructPoint3ds;
private:
   void PointTrack(ImagePtr& Image);
   void computeTrack(std::vector<ImagePtr>&Images);
   void InitTrackPointEdge(ImagePtr& image1,ImagePtr& image2);
   void createEdge(std::vector<ImagePtr>&Images);
   void selectImagePair(std::vector<ImagePtr>& Images);
   void findNextEdge(std::vector<ImagePtr>&Images,ImagePtr& alreadyImage,ImagePtr& nextImage);
   void ImagePairTrigulate(ImagePtr& image1,ImagePtr& image);
  
   void addReconstructImage(ImagePtr& Image);
   void addReconstructPoint3d(Point3dPtr& Point3d);
   void filterPoint();
   void filterImage();
   int trackTheshold_;
   int edgeThreshold_;
   int candiateImagePair_;
   int minAngleTheshold_;
   int maxAngleTheshold_;
   double obervationProjectError_;
   double filterProjectError_;
   std::string savePointCloudPath_;
   std::string savePointPath_;
   std::string savePosePath_;
 
   std::unordered_set<ImagePtr,HashImage<ImagePtr>,EqualImage<ImagePtr>>reconstructImagesSet;
  
   std::unordered_set<Point3dPtr,HashPoint3d<Point3dPtr>,EqualPoint3d<Point3dPtr>>reconstructPoint3dsSet;
};
#endif