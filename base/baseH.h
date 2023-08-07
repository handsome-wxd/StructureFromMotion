#ifndef BASE_H
#define BASE_H
#pragma once
//此头文件用来包含一些系统中通用头文件,以及通用函数
#include<iostream>
#include<vector>
#include<unordered_set>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#define PI acos(-1)

struct HashCVPoint2d{
    size_t operator()(const cv::Point2d& p)const{
        return std::hash<float>()(p.x)^std::hash<float>()(p.y);
    }
};
struct HashCVPoint3d{
    size_t operator()(const cv::Point3d& p)const {
        return std::hash<float>()(p.x)^std::hash<float>()(p.y)^std::hash<float>()(p.z);
    }
};


#endif