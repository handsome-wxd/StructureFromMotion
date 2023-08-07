#ifndef EVALPROJECTION_H
#define EVALPROJECTION_H
#include "../base/baseH.h"
#include "../base/Image.h"
class evalProjection{

public:
     evalProjection(std::vector<Point3dPtr>&point3ds):point3ds_(point3ds){}
     void errorReport();
private:
    void computeError();
    void errorStatistics();
    std::vector<Point3dPtr>point3ds_;
    std::vector<double>xError;
    std::vector<double>yError;
};
#endif