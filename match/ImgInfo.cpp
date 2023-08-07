#include "ImgInfo.h"
#include "../base/baseH.h"

bool is_file(const std::string& filename) {
        struct stat   buffer;
        return (stat (filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
    }
void getFileNames(const std::string& path,std::vector<std::string>& filenames){
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            if(is_file(path + "/" + ptr->d_name))
                 filenames.emplace_back(path + "/" + ptr->d_name);
        }   
    }
    closedir(pDir);
}


void InputImageInformaton::ImageLoad(){
    std::vector<std::string>fileAddres;
    getFileNames(ImagesPath_,fileAddres);
    int index=0;
    for(std::string fileAddr:fileAddres){
        cv::Mat brgImg=cv::imread(fileAddr,cv::IMREAD_COLOR);
        cv::Mat grayImg=cv::imread(fileAddr,cv::IMREAD_GRAYSCALE);
        if(brgImg.empty() || grayImg.empty())
            continue;
        ImagePtr image(new Image(fileAddr,index,brgImg,grayImg));
        Images.emplace_back(image);
        index+=1;
    }
    return ;
}

int readImg(const std::string &inputPath,std::vector<cv::Mat>&ImgData){
    std::vector<std::string>fileAddrs;
    getFileNames(inputPath,fileAddrs);
    // vector<cv::Mat>data;
    int index=0;
    for(std::string fileAddr:fileAddrs){
        cv::Mat img=cv::imread(fileAddr,cv::IMREAD_COLOR);
        if(img.data==nullptr)
            return -1;
        ImgData.emplace_back(img);
    }
    return 0;
}


void InputImageInformaton:: evalEvalIntrisic(const std::string& filePath,cv::Mat& Intrinsic){
    std::vector<cv::Mat>imgs;
    int ret=readImg(filePath,imgs);
    int boardWidth = 7;  // 棋盘格横向内角点数量
    int boardHeight = 7; // 棋盘格纵向内角点数量
    float squareSize = 1.f; // 棋盘格格子的大小，单位为米,随便设置，不影响相机内参计算
    cv::Size boardSize(boardWidth, boardHeight);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> corners;

    // 2. 拍摄棋盘图像
    cv::Mat image, gray;
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    for (size_t i = 0; i < imgs.size(); i++)
    {
        image=imgs[i];
        cvtColor(image, gray,cv::COLOR_BGR2GRAY);
        bool found =cv::findChessboardCorners(image, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (found)
        {
            cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(image, boardSize, corners, found);
         

            std::vector<cv::Point3f> objectCorners;
            for (int j = 0; j < boardHeight; j++)
            {
                for (int k = 0; k < boardWidth; k++)
                {
                    objectCorners.push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
                }
            }
            objectPoints.push_back(objectCorners);
            imagePoints.push_back(corners);
        }
    }
    // 4. 标定相机
    cv::Mat distCoeffs;//畸变系数
    std::vector<cv::Mat> rvecs, tvecs;
    calibrateCamera(objectPoints, imagePoints, image.size(), Intrinsic, distCoeffs, rvecs, tvecs);
    return ;
}


void InputImageInformaton::readK(const std::string& filePath,cv::Mat &Matrix){
    std::ifstream infile;
    Matrix.create(3,3,CV_64FC1);
    infile.open(filePath);
    if (!infile)
    {
        std::cout << "无法打开内参文件文件！" << std::endl;
        exit(1);
    }
    std::string value;
    for(int i=0;i<3;++i){
        double* ptr=Matrix.ptr<double>(i);
        for(int j=0;j<3;++j){
            infile>>value;
            // cout<<value<<endl;
            ptr[j]=stoi(value);
        }
    }
}


void InputImageInformaton::LoadImageInfo(){
    ImageLoad();
    cv::Mat Intrinsic;
    if(isEvalIntrisic_){
         evalEvalIntrisic(IntrinsicsPath_,Intrinsic);
    }
    else{
        readK(IntrinsicsPath_,Intrinsic);
    }
    //对图像初始化内参
    for(auto imgPtr:Images){
        imgPtr->Intrinsics_=Intrinsic;
    }  
    return ;
}