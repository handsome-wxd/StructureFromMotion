#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <list>
#include "locker.h"

template<typename T>
class Threadpool{
    public:
    Threadpool(int thread_number=1,int max_request=100);
    ~Threadpool();
    bool append(T request);
    bool workEmpty();
    int workSize();
    void waitAllTasks();
    private:
        static void* worker(void *arg);
        void run();
    private:
        int thread_number_;
        int max_requests_;
        pthread_t *threads_;
        std::list<T>workqueue_;
        locker queuelocker_;
        sem queuestat_;
};




template<typename T>
Threadpool<T>::Threadpool(int thread_number,int max_request):thread_number_(thread_number),max_requests_(max_request){
    if(thread_number<=0 || max_request<=0){
        throw std::exception();
    }
    threads_=new pthread_t[thread_number_];
    if(!threads_){
        throw std::exception();
    }
    for(int i=0;i<thread_number_;++i){
        if(pthread_create(threads_+i,NULL,worker,this)!=0){
            delete []threads_;
             throw std::exception();
        }
        //  pthread_join(threads_+i, NULL); 
    }
}
template<typename T>
Threadpool<T>::~Threadpool(){
    delete[] threads_;
}

template<typename T>
void Threadpool<T>::waitAllTasks(){
    while(!workEmpty()){
        std::cout<<"";
    }
}

template<typename T>
bool Threadpool<T>::append(T request){
    queuelocker_.lock();
    // std::cout<<"加入任务"<<std::endl;
    if(workqueue_.size()>=max_requests_){
        queuelocker_.unlock();
        return false;
    }
    workqueue_.push_back(request);
    queuelocker_.unlock();
    queuestat_.post();
    return true;
}

template<typename T>
bool Threadpool<T>:: workEmpty(){
    return  workqueue_.empty();
}
template<typename T>
int Threadpool<T>:: workSize(){
    return  workqueue_.size();
}
template<typename T>
void *Threadpool<T>::worker(void *arg){
    Threadpool *pool=(Threadpool *)arg;
    pool->run();
    return pool;
}
template<typename T>
void Threadpool<T>::run()
{
    while(true){
        queuestat_.wait();
        queuelocker_.lock();
        if(workqueue_.empty()){
           queuelocker_.unlock();
            break;
        }
        T request=workqueue_.front();
        workqueue_.pop_front();
        queuelocker_.unlock();
        if(!request){
            continue;
        }
        request->work();
    }
}
#endif
// #include <iostream>
// #include <thread>
// #include <queue>
// #include <functional>
// template<typename T>
// class ThreadPool {
// public:
//     ThreadPool(size_t numThreads) {
//         for (size_t i = 0; i < numThreads; ++i) {
//             threads.emplace_back([=] {
//                 while (true) {
//                     std::function<void()> task;
//                     {
//                         std::unique_lock<std::mutex> lock(queueMutex);
//                         condition.wait(lock, [=] { return stop || !tasks.empty(); });
//                         if (stop && tasks.empty()) {
//                             return;
//                         }
//                         task = std::move(tasks.front());
//                         tasks.pop();
//                     }
//                     task();
//                 }
//             });
//         }
//     }

//     template <class F, class... Args>
//     void enqueue(F&& f, Args&&... args) {
//         {
//             std::unique_lock<std::mutex> lock(queueMutex);
//             tasks.emplace([=] { return f(args...); });
//         }
//         condition.notify_one();
//     }

//     ~ThreadPool() {
//         {
//             std::unique_lock<std::mutex> lock(queueMutex);
//             stop = true;
//         }
//         condition.notify_all();
//         for (std::thread& thread : threads) {
//             thread.join();
//         }
//     }

// private:
//     std::vector<std::thread> threads;
//     std::queue<std::function<void()>> tasks;
//     std::mutex queueMutex;
//     std::condition_variable condition;
//     bool stop = false;
// };

// // 用法示例
// template<typename T>
// void exampleTask(T function) {
//     function->work();
//     std::cout << "Task executed with value: " << value << std::endl;
// }

// int main() {
//     ThreadPool pool(4);  // 创建4个线程的线程池
//     // 向线程池中添加任务
//     for (int i = 0; i < 10; ++i) {
//         pool.enqueue(exampleTask, i);
//     }
//     // 一段时间后销毁线程池
//     std::this_thread::sleep_for(std::chrono::seconds(2));
//     return 0;
// }

// #endif