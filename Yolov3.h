#ifndef YOLOV3_H_
#define YOLOV3_H_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <string>
#include <cstring>

#include "NvInfer.h"
#include "NvOnnxParserRuntime.h"

#include <npp.h>

#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>

#include "result_process.h"
#include "Logger.h"
#include "result_process.h"
using namespace std;
using namespace nvinfer1;

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

class Yolov3{
private:
    const int batch_size;
    int INPUT_H;
    int INPUT_W;
    int OUTPUT_SIZE;
    
    void* buffers[4];
    vector<int> buf_size;
    
    void *gpu_img_buf, *gpu_img_resize_buf, *gpu_data_buf;
    void * gpu_data_planes;
    cudaStream_t stream;

    ICudaEngine* engine;
    IExecutionContext* context;
    IRuntime* runtime;
    Logger g_logger;
    const int g_use_dla_core;
    float * result;
    vector<string> cls_names;
    
    void get_class_names(const char * const file);

    yolov3_result_parser parser1, parser2, parser3;

    void print_boxes(vector<box>& box_list) const;
    static int nms_box_compare(const void* a, const void* b);
    void sort_boxlist(vector<box>& box_list) const;
    

    int result_step1, result_step2;
    const float obj_thresh;
    const float nms_thresh; 

    void make_input(const cv::Mat &img);
    void make_yolov3_result(vector<box>& box_list);
    int nms(vector<box>& box_list, vector<box>& box_result) const;
public:
    Yolov3(const int img_w, const int img_h, const float *anchors, const int mask[][3], 
                const float obj_thresh, const float nms_thresh, const char *trt_file, 
                const int cls_num = 1, const char *cls_name_file = "./person_label.txt", 
                const int batch_size = 1, const int g_use_dla_core = -1);
    
    ~Yolov3();
    void doInference(const cv::Mat &img, vector<box>& box_list, vector<box>& box_result);
    void make_output(cv::Mat &img, vector<box> &box_list) const;
    
};

#endif
