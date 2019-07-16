#ifndef __DEEPLAB_CUH__
#define __DEEPLAB_CUH__

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <time.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <npp.h>
#include "Logger.h"

using namespace nvinfer1;
using namespace plugin;

#define MAX_IM (2000*2000)

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)



__global__ void prob2rgb_kernel(const float * const prob, unsigned char * const rgb, const int N);
class DeepLab
{
public:
    DeepLab(const int img_w, const int img_h, const char *trt_file, const int batch_size = 1, const int threads_per_block = 4, int g_use_dla_core = -1);
    ~DeepLab();
    void doInference(const cv::Mat& im);
    cv::Mat real_out;
    int get_OUTPUT_W() const;
    int get_OUTPUT_H() const;
private:
    void prob2rgb();
    const int batch_size;
    const int g_use_dla_core;
    int cls_num;
    Dims3 input_dims;
    Dims3 output_dims;
    size_t input_size;
    size_t output_size;

    dim3 blocks;
    dim3 threads;
    IRuntime* infer;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    void *gpu_img_buf, *gpu_img_resize_buf, *gpu_data_buf;
    void * gpu_data_planes;
    int INPUT_W;
    int INPUT_H;
    int OUTPUT_W;
    int OUTPUT_H;
    const int threads_per_block;
    Logger g_logger;
    void* buffers[2];
    void *rgb;
    void make_input(const cv::Mat &img);
};

#endif
