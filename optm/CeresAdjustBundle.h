#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include<ceres/ceres.h>
#include<chrono>
#include "../base/Image.h"
#include "../base/Point3d.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y,double _f1,double _f2,double _cx,double _cy) : 
    observed_x(observation_x),observed_y(observation_y),f1(_f1),f2(_f2),cx(_cx),cy(_cy) {}

    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjection(camera, point, predictions,f1,f2,cx,cy);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjection(const T *camera, const T *point, T *predictions,double f1,double f2,double cx,double cy) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        T xp=p[0]/p[2];
        T yp=p[1]/p[2];
        predictions[0] = T(f1)*xp+ T(cx);
        predictions[1] = T(f2)*yp+ T(cy);

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y,const double f1,const double f2,double cx,double cy) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2,6, 3>(
            new SnavelyReprojectionError(observed_x, observed_y,f1,f2,cx,cy)));
    }

private:
    double observed_x;
    double observed_y;
    double f1;
    double f2;
    double cx;
    double cy;
};


void poseToVec(std::vector<ImagePtr>&Images,double *ptr, int cameraBlockSize);

//向量转相机位姿
void vecToPose(double *ptr,std::vector<ImagePtr>&Images, int cameraBlockSize);


//将点转化为一维向量
void point3dToVec(std::vector<Point3dPtr>&Point3ds,double *pointPtr, int pointBlockSize );
//一维向量转3d点

void vecToPoint3d(double *pointPtr,std::vector<Point3dPtr>&oint3ds, int pointBlockSize );

/***
 * 输入相机位姿，3d点的坐标
*/
void SolveCeresBA(std::vector<ImagePtr>& Images,std::vector<Point3dPtr>& Point3ds);
#endif // SnavelyReprojection.h



