#ifndef CAMERAUTILS_H
#define CAMERAUTILS_H
#include <opencv2/opencv.hpp>
#include <cmath>
#include "CameraParameter.h"
float disparityDepthConversion(float f,float baseline,float d){
    return f*baseline/d;
}
cv::Mat_<float> scaleK(cv::Mat_<float>K,float scaleFactor){
    cv::Mat_<float>KScaled=K.clone();
    KScaled(0,0)=K(0,0)/scaleFactor;
    KScaled(1,1)=K(1,1)/scaleFactor;
    KScaled(0,2)=K(0,2)/scaleFactor;
    KScaled(1,2)=K(1,2)/scaleFactor;
    return KScaled;
}
float VecNormL2(cv::Vec3f m,cv::Vec3f n){
     float squareSum=(m(0)-n(0))*(m(0)-n(0))+
                      (m(1)-n(1))*(m(1)-n(1))+
                      (m(2)-n(2))*(m(2)-n(2));
      return std::sqrt(squareSum);
}
float MatNormL2(cv::Mat_<float>m,cv::Mat_<float>n){
    if(m.size()!=n.size()){
        std::cout<<"输入矩阵尺寸不一致"<<std::endl;
    }
    float squareSum=0.f;
    for(int row=0;row<m.rows;++row){
        for(int col=0;col<m.cols;++col){
            squareSum+=(m(col,row)-n(col,row))*(m(col,row)-n(col,row));
        }
    }
    return std::sqrt(squareSum);
}
void copyOpencvMatToFloatArray(cv::Mat_<float>&m,float **arr){
    for(int pj=0;pj<m.rows;++pj){
        for(int pi=0;pi<m.cols;++pi){
            (*arr)[pi+pj*m.cols]=m(pj,pi);
        }
    }
}
void copyOpencvVecToFloat4(cv::Vec3f &v,float4* a){
    a->x=v(0);
    a->y=v(1);
    a->z=v(2);
    return ;
}
cv::Vec3f getViewVector(Camera camera,float u,float v,float depth){
    cv::Mat_<float>pt=cv::Mat::ones(3,1,CV_32F);
    pt(0,0)=u;
    pt(0,1)=v;
   
    //K(R*X+t)=x X=(K*R).inv*(x-k*t)
    //X到光心的方向 V=X-C
    cv::Mat_<float>ptX=camera.M_inv*(depth*pt-camera.P.col(3));
    return cv::Vec3f(ptX(0,0)-camera.Center(0),ptX(1,0)-camera.Center(1),ptX(2,0)-camera.Center(2));
    // return v;

}
static float getAngle ( cv::Vec3f v1, cv::Vec3f v2 ) {
	float angle = acosf ( v1.dot ( v2 ) );
	return angle;
}
cv::Mat_<float> getTransformationMatrix ( cv::Mat_<float> R, cv::Mat_<float> t ) {
    cv::Mat_<float> transMat = cv::Mat::eye ( 4,4, CV_32F );
    //Mat_<float> Rt = - R * t;
    R.copyTo ( transMat ( cv::Range ( 0,3 ),cv::Range ( 0,3 ) ) );
    t.copyTo ( transMat ( cv::Range ( 0,3 ),cv::Range ( 3,4 ) ) );

    return transMat;
}
cv::Mat_<float> getTransformationReferenceToOrigin ( cv::Mat_<float> R,cv::Mat_<float> t ) {
    cv::Mat_<float> transMat_original = getTransformationMatrix ( R,t );
    return transMat_original.inv ();
}
void transformCamera ( cv::Mat_<float> R,cv::Mat_<float> t, cv::Mat_<float> transform, Camera &cam, cv::Mat_<float> K ) {
    cv::Mat_<float> transMat_original = getTransformationMatrix ( R,t );
    cv::Mat_<float> transMat_t = transMat_original * transform;

    //计算投影矩阵
    cam.P = K * transMat_t ( cv::Range ( 0,3 ),cv::Range ( 0,4 ) );
    // set R and t
    cam.R = transMat_t ( cv::Range ( 0,3 ),cv::Range ( 0,3 ) );
    cam.t = transMat_t ( cv::Range ( 0,3 ),cv::Range ( 3,4 ) );
    // set camera center C
    cv::Mat_<float> C =-cam.R.inv()*cam.t;
    cam.Center=cv::Vec3f(C(0,0),C(1,0),C(2,0));
}

enum DecompTypes {
    /** Gaussian elimination with the optimal pivot element chosen. */
    DECOMP_LU       = 0,
    /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
    src1 can be singular */
    DECOMP_SVD      = 1,
    /** eigenvalue decomposition; the matrix src1 must be symmetrical */
    DECOMP_EIG      = 2,
    /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
    defined */
    DECOMP_CHOLESKY = 3,
    /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
    DECOMP_QR       = 4,
    /** while all the previous flags are mutually exclusive, this flag can be used together with
    any of the previous; it means that the normal equations
    \f$\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}\f$ are
    solved instead of the original system
    \f$\texttt{src1}\cdot\texttt{dst}=\texttt{src2}\f$ */
    DECOMP_NORMAL   = 16
};
#endif
