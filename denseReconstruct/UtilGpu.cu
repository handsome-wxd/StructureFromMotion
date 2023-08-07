#include "UtilGpu.h"
void selectCudaDevice()
{
    int deviceCount=0;
    checkCudaError(cudaGetDeviceCount(&deviceCount));
    if(deviceCount==0){
        fprintf(stderr,"There is no cuda capable device!\n");
        exit(EXIT_FAILURE);
    }
    std::cout<<"Detected"<<deviceCount<<"devices!"<<std::endl;
    std::vector<int>usableDevices;
    std::vector<std::string>usableDeviceNames;
    for(int i=0;i<deviceCount;++i){
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
            //prop.major表示GUP的计算能力主版本号，比如8，prop.minor表示计算能力次版本号比如4，和起来就是8.4
            if(prop.major>=3 && prop.minor>=0){
                usableDevices.push_back(i);
                usableDeviceNames.push_back(prop.name);
            }
            else{
                std::cout<<"CUDA capable device"<<std::string(prop.name)
                <<"is only compute cabablilty"<<prop.major<<"."<<prop.minor<<std::endl;
            }
        }
        else{
            std::cout<<"Could not check device properties for one for the cuda devices!"<<std::endl;
        }
    }
    if(usableDevices.empty()){
        fprintf(stderr,"There is no cuda device support!\n");
        exit(EXIT_FAILURE);
    }
    std::cout<<"Detected code compatible device: "<<usableDeviceNames[0]<<std::endl;
    checkCudaError(cudaSetDevice(usableDevices[0]));
    checkCudaError(cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*128));
}