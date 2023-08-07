#include "DenseMatch.h"
#include <algorithm>
#include "CameraUtils.h"

AlgorithmParameters* algParams;
GlobalState* globalState;


CameraParameters setCameraParameterscu(CameraParameters_cu& CP,std::vector<ImagePtr>& Images_,const int reference,float scaleFactor,bool transformP){
   CameraParameters camParams;
   size_t numCameras=Images_.size();
   camParams.cameras.resize(numCameras);
   std::vector<cv::Mat_<float>>k(numCameras);
   std::vector<cv::Mat_<float>>R(numCameras);
   std::vector<cv::Mat_<float>>T(numCameras);
   std::vector<cv::Mat_<float>>C(numCameras);
   //std::vector<cv::Mat_<float>>center(numCameras);
   for(size_t i=0;i<numCameras;++i){
        if(Images_[i]->Intrinsics_.type()!=CV_32F){
            Images_[i]->Intrinsics_.convertTo(Images_[i]->Intrinsics_,CV_32F);
            Images_[i]->R_.convertTo(Images_[i]->R_,CV_32F);
            Images_[i]->T_.convertTo(Images_[i]->T_,CV_32F);
            //Images_[i]->cameraCpu_->OptionCenter_.convertTo(Images_[i]->cameraCpu_->OptionCenter_,CV_32F);
        }
        k[i]=Images_[i]->Intrinsics_;
        R[i]=Images_[i]->R_; 
        T[i]=Images_[i]->T_;
        //center[i]=Images_[i]->cameraCpu_->OptionCenter_;
   }
   camParams.cameras[reference].reference=true;
   camParams.idRef=reference;
   camParams.k=scaleK(k[0],scaleFactor);
   camParams.k_inv=camParams.k.inv();
   camParams.f=camParams.k(0,0);
    //将所有图像的相机参数，转到以reference图像为中心的世界坐标下
    // cv::Mat referenceTranform;
    // cv::hconcat(R[reference],T[reference],referenceTranform);
   cv::Mat_<float>referenceTranform=getTransformationReferenceToOrigin(R[reference],T[reference]);
  //std::cout<<"referenceTranform="<<referenceTranform<<std::endl;
   for(size_t i=0;i<numCameras;++i){
        camParams.cameras[i].k=scaleK(k[i],scaleFactor);
        camParams.cameras[i].k_inv=camParams.cameras[i].k.inv();
        camParams.cameras[i].R_orig_inv=R[i].inv();
        transformCamera(R[i],T[i],referenceTranform,camParams.cameras[i],camParams.k);
        std::cout<<"R"<<i<<"="<<camParams.cameras[i].R<<std::endl;
        std::cout<<"t"<<i<<"="<<camParams.cameras[i].t<<std::endl;
        // camParams.cameras[i].t= T[i];
        // cv::Mat_<float>RT;
        // cv::hconcat(R[i], T[i],RT) 
        // camParams.cameras[i].P=camParams.cameras[i].k*RT;
        // cv::vconcat(camParams.cameras[i].P, cv::Mat::zeros(1,4,CV_32F), camParams.cameras[i].P);
        // camParams.cameras[i].P(3,3)=1.0f;
        // std::cout<<"P_inv"<<i<<"="<<camParams.cameras[i].P<<std::endl;
        camParams.cameras[i].P_inv=camParams.cameras[i].P.inv(DECOMP_SVD);
        camParams.cameras[i].M_inv=camParams.cameras[i].P.colRange(0,3).inv();
        camParams.cameras[i].baseline=algParams->baseline;
        // std::cout<<"P_inv ="<<camParams.cameras[i].P_inv<<std::endl;
        // std::cout<<"M_inv ="<<camParams.cameras[i].M_inv<<std::endl;
        // std::cout<<"k_inv ="<<camParams.cameras[i].k_inv<<std::endl;
        copyOpencvMatToFloatArray(camParams.cameras[i].k,&CP.cameras[i].k);
        copyOpencvMatToFloatArray(camParams.cameras[i].k_inv,&CP.cameras[i].k_inv);
        copyOpencvMatToFloatArray(camParams.cameras[i].R_orig_inv,&CP.cameras[i].R_orig_inv);
        CP.cameras[i].fy=camParams.k(1,1);
        CP.f=camParams.k(0,0);
        CP.cameras[i].f=camParams.k(0,0);
        CP.cameras[i].fx=camParams.k(0,0);
        CP.cameras[i].fy=camParams.k(1,1);
        CP.cameras[i].baseline=camParams.cameras[i].baseline;
        CP.cameras[i].reference=camParams.cameras[i].reference;
        CP.cameras[i].alpha=camParams.k(0,0)/camParams.k(1,1);
        copyOpencvMatToFloatArray(camParams.cameras[i].P,&CP.cameras[i].P);
        copyOpencvMatToFloatArray(camParams.cameras[i].P_inv,&CP.cameras[i].P_inv);
        copyOpencvMatToFloatArray(camParams.cameras[i].M_inv,&CP.cameras[i].M_inv);
        copyOpencvMatToFloatArray(camParams.cameras[i].R,&CP.cameras[i].R);
        copyOpencvVecToFloat4(camParams.cameras[i].Center,&CP.cameras[i].C4);
        CP.cameras[i].t4.x=camParams.cameras[i].t(0,0);
        CP.cameras[i].t4.y=camParams.cameras[i].t(1,0);
        CP.cameras[i].t4.z=camParams.cameras[i].t(2,0);
        cv::Mat_<float>tmp=camParams.cameras[i].P.col(3);
        CP.cameras[i].P_col34.x=tmp(0,0);
        CP.cameras[i].P_col34.y=tmp(1,0);
        CP.cameras[i].P_col34.z=tmp(1,0);
   }
   return camParams;
}
void selectViews(CameraParameters& cameraParams,int imgWidth,int imgHight,AlgorithmParameters& algParams){
    std::vector<Camera>&cameras=cameraParams.cameras;
    printf("ref=%d\n",cameraParams.idRef);
    Camera ref=cameras[cameraParams.idRef];
    int u=imgWidth/2;
    int v=imgHight/2;
    cameraParams.viewSelectionSubset.clear();
    cv::Vec3f viewVectorRef=getViewVector(ref,u,v,1.0f);
    //printf("viewVectorRef=%f,%f,%f",viewVectorRef[0],viewVectorRef[1],viewVectorRef[2]);
    //printf("algParams.minAngle=%f,algParams.maxAngle=%f\n",algParams.minAngle,algParams.maxAngle);
    float minnum_angle_radians=algParams.minAngle*M_PI/180.f;
    float maxnum_angle_radians=algParams.maxAngle*M_PI/180.f;
    int max_views=algParams.maxViews;
    float min_depth=9999;
    float max_depth=0;
    // printf("cameraParams.idRef=%d\n",cameraParams.idRef);
    float sumBaseline;
    if(algParams.viewSelection){
        for(size_t i=0;i<cameras.size();++i){
            if(i==cameraParams.idRef)
                continue;
            cv::Vec3f vec=getViewVector(cameras[i],u,v,1.0f);
            // std::cout<<"ref,center="<<ref.Center<<std::endl;
            // std::cout<<"camera,center="<<cameras[i].Center<<std::endl;

            float baseline=VecNormL2(ref.Center,cameras[i].Center);
            sumBaseline+=baseline;
            float angle=getAngle(viewVectorRef,vec);
            //printf("angle=%f\n",angle);
            if(angle>minnum_angle_radians && angle<maxnum_angle_radians){
                if(algParams.viewSelection){
                    cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));
                }
                float min_range=(baseline/2.0f)/sin(maxnum_angle_radians/2.0f);
                float max_range=(baseline/2.0f)/sin(minnum_angle_radians/2.0f);
                min_depth=std::min(min_depth,min_range);
                max_depth=std::max(max_depth,max_range);
            }
        }
        if(algParams.depthMin==-1.0f){
            if(min_depth==9999)
                min_depth=0;
            algParams.depthMin=min_depth;
        }
        if(algParams.depthMax==-1.0f){
            if(max_depth==0)
                max_depth=20;
            algParams.depthMax=max_depth;
        }
        //printf(" algParams.depthMin=%f, algParams.depthMax=%f\n",algParams.depthMin, algParams.depthMax);
    }
    else{
        for(size_t i=0;i<cameras.size();++i){
            cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));
        }
        return ;
    }
   
    if(cameraParams.viewSelectionSubset.size()>=max_views){
        std::srand(unsigned(std::time(0)));
        std::random_shuffle(cameraParams.viewSelectionSubset.begin(),cameraParams.viewSelectionSubset.end());
        cameraParams.viewSelectionSubset.erase(cameraParams.viewSelectionSubset.begin()+max_views,cameraParams.viewSelectionSubset.end());
    }
    if(sumBaseline!=0){
        // printf("sumBaseline=%f\n",sumBaseline);
        algParams.baseline=sumBaseline/cameraParams.viewSelectionSubset.size();
    }
    
  
}
void  addImageToTextureFloatColor (std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        //  printf("rows=%d,cols=%d\n",rows,cols);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
       checkCudaError(cudaMallocArray(&cuArray[i],&channelDesc,cols,rows));
        checkCudaError(cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float)*4,
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaError(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
    return;
}

void addImageToTextureFloatGray (std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
       
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc (32,
                               0,
                               0,
                               0,
                               cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        checkCudaError(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaError (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaError(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}

void PreconditioningGPUdata(std::vector<ImagePtr>&Images_,const int reference){
    if(Images_.size()<2){
        std::cout<<"输入视图个数必须大于2"<<std::endl;
        exit(-1);
    }
    size_t numImages=Images_.size();
    algParams=new AlgorithmParameters();
    globalState=new GlobalState();
    globalState->reference=reference;
    algParams->numImgProcessed=std::min((int)numImages,algParams->numImgProcessed);
    CameraParameters cameraParameters=setCameraParameterscu(*(globalState->cameras),Images_,reference);
    selectViews(cameraParameters,Images_[reference]->cols,Images_[reference]->rows,*algParams);
    int numSelectViews=cameraParameters.viewSelectionSubset.size();
    for(int i=0;i<numSelectViews;++i){
        globalState->cameras->viewSelectionSubset[i]=cameraParameters.viewSelectionSubset[i];
    }
    for(int i=0;i<algParams->numImgProcessed;++i){
        cameraParameters.cameras[i].depthMin=algParams->depthMin;
        cameraParameters.cameras[i].depthMax=algParams->depthMax;
        cameraParameters.cameras[i].baseline=algParams->baseline;
        algParams->minDisparity=disparityDepthConversion(cameraParameters.f,cameraParameters.cameras[i].baseline,cameraParameters.cameras[i].depthMax);
        algParams->minDisparity=disparityDepthConversion(cameraParameters.f,cameraParameters.cameras[i].baseline,cameraParameters.cameras[i].depthMin);
    }
    globalState->params=algParams;
    globalState->cameras->viewSelectionSubsetNumber=static_cast<int>(numSelectViews);
    globalState->cameras->cameras->baseline=algParams->baseline;
    globalState->cameras->cols=Images_[0]->cols;
    globalState->cameras->rows=Images_[0]->rows;
    globalState->params->cols=Images_[0]->cols;
    globalState->params->rows=Images_[0]->rows;
    globalState->lines->n=Images_[0]->cols*Images_[0]->rows;
    globalState->lines->mallocSpace(globalState->lines->n);
    globalState->lines->length=Images_[0]->cols;
    globalState->lines->stride=Images_[0]->rows;

    std::vector<cv::Mat > img_grayscale_float(numImages);
    std::vector<cv::Mat > img_color_float(numImages);
    std::vector<cv::Mat > img_color_float_alpha(numImages);
    std::vector<cv::Mat_<uint16_t> > img_grayscale_uint(numImages);
    for (size_t i = 0; i<numImages; i++)
    {
        Images_[i]->grayImage_.convertTo(img_grayscale_float[i], CV_32FC1); // or CV_32F works (too)
        Images_[i]->grayImage_.convertTo(img_grayscale_uint[i], CV_16UC1); // or CV_32F works (too)
        if(algParams->colorProcessing) {
            std::vector<cv::Mat_<float> > rgbChannels ( 3 );
            img_color_float_alpha[i] = cv::Mat::zeros (  Images_[i]->rows,  Images_[i]->cols, CV_32FC4 );
            Images_[i]->rgbImage_.convertTo (img_color_float[i], CV_32FC3); // or CV_32F works (too)
            cv::Mat alpha( Images_[i]->rows, Images_[i]->cols, CV_32FC1 );
            cv::split (img_color_float[i], rgbChannels);
            rgbChannels.push_back( alpha);
            cv::merge (rgbChannels, img_color_float_alpha[i]);
        }
    }
   
    if(algParams->colorProcessing){
        addImageToTextureFloatColor( img_color_float_alpha,globalState->imgs,globalState->cuArray);
    }
    else{
        addImageToTextureFloatGray( img_grayscale_float,globalState->imgs,globalState->cuArray);
    }
    return ;
}
void patchMatch(){
    pathMatchStereo(0,*globalState);
    return ;
}
void saveImgDepth(ImagePtr referenceImg,GlobalState& globalState ){
    int rows=referenceImg->rows;
    int cols=referenceImg->cols;
    cv::Mat_<float>cudaDepth=cv::Mat::zeros(rows,cols,CV_32FC1);
    // referenceImg->DepthMap=cv::Mat::zeros(rows,cols,CV_32FC1);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            int center=i*cols+j;
            //printf("depth=%f",globalState.lines->norm4[center].w);
            cudaDepth.at<float>(i,j)=globalState.lines->norm4[center].w;
        }
    }
    referenceImg->DepthMap=cudaDepth;
    cv::Mat normalizedImage;
    cv::normalize(cudaDepth,normalizedImage, 0, 65536, cv::NORM_MINMAX,CV_16U);  // 归一化图像到0-255范围
    // std::string fileName=
    cv::imwrite("output"+std::to_string(referenceImg->imageIndex_)+".png", normalizedImage);
    // cv::Mat normalizedImage;
    // depthMap.convertTo(normalizedImage, CV_8U, 255.0);
    // cv::imwrite("depth.png",normalizedImage);
}
void DenseMatch(std::vector<ImagePtr>&Images_){
    selectCudaDevice();
    for(int i=0;i<Images_.size();++i){
        PreconditioningGPUdata(Images_,i);
        patchMatch();
        saveImgDepth(Images_[i],*globalState);
        delete globalState;
        delete algParams;
    }
    return ;
    
}
