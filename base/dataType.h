#ifndef DATATYPE_H
#define DATATYPE_H
#include <Eigen/Core>
//定义数据的基本类型
typedef double PointType;

typedef int ImageIndexType;
typedef std::string ImageNameType;
typedef double MatrixType;
typedef Eigen::Matrix<MatrixType,3,3> Matrix3d;
typedef Eigen::Matrix<MatrixType,3,4> Matrix34d;
typedef Eigen::Matrix<MatrixType,3,1> Matrix31d;
#endif