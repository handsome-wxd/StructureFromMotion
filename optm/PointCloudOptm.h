#ifndef POINTCLOUNDOPTM_H
#define POINTCLOUNDOPTM_H
#include "../base/baseH.h"
#include "../base/Image.h"
class PointCloundOptm{
    public:
        PointCloundOptm(double xErrorThreshold,double yErrorThreshold):xErrorThreshold_(xErrorThreshold),yErrorThreshold_(yErrorThreshold){}
        void filterErrorPoint(std::vector<Point3dPtr>& point3ds,std::vector<Point3dPtr>& filterPoint3ds);

    private:
        double xErrorThreshold_;
        double yErrorThreshold_;
};

#endif