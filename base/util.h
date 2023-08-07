#include "baseH.h"
#include "Image.h"

double getAngelOfTwoVector(std::vector<double>line1,std::vector<double>line2);
double ImagePairAngle(cv::Mat Point3d,cv::Mat& opticsCenter1,cv::Mat& opticsCenter2);
void savePointCloud(std::vector<Point3dPtr>&reconstructPoint3ds,std::string &savePath);
void saveTxt(std::vector<ImagePtr>&Images,std::vector<Point3dPtr>&Point3d,std::string& savePointPath,std::string& savePosePath);
void createPath(std::string& path);
bool evalObservationDepthAndProject(Point2dPtr& point2d,double threshold);
bool isValildIntrisicMatrix(cv::Mat& Intrinsics,int rows,int cols);
bool isValildRotatedMatrix(cv::Mat& Intrisic);


