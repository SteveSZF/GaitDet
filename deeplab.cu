#include "deeplab.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <NvOnnxParserRuntime.h>
#include "Logger.h"

const int num_cls = 2;
__constant__ unsigned char map_[num_cls][3] = { {0, 0, 0},
                            {255, 255, 255} };

//transfer the prob map to the rgb image on the GPU
__global__ void prob2rgb_kernel(const float * const prob, unsigned char * const rgb, const int N){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = x + y * blockDim.x * gridDim.x;
	int maxIdx = 0;
	if(tid < N){
        for(int i = 1; i < num_cls; i++){
		    if(prob[tid + i * N] > prob[tid + maxIdx * N])
			maxIdx = i;
		}
		
	}
	rgb[tid * 3] = map_[maxIdx][2];
	rgb[tid * 3 + 1] = map_[maxIdx][1];
	rgb[tid * 3 + 2] = map_[maxIdx][0];
}
DeepLab::DeepLab(const int img_w, const int img_h, const char *trt_file, const int batch_size, const int threads_per_block, const int g_use_dla_core)
                :batch_size(batch_size), threads_per_block(threads_per_block), g_use_dla_core(g_use_dla_core)
{
    //read the size of trt file and read the trt file
    uchar* buf;
    FILE* fp = fopen(trt_file, "rb");
    fseek(fp, 0L, SEEK_END); 
    int size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    buf = (unsigned char*)malloc(size);
    fread((void*)buf, 1, size, fp);
    fclose(fp);
    
    //generate the engine and the context for inference
    IPluginFactory* plugin = nvonnxparser::createPluginFactory(g_logger);
    assert(plugin != nullptr);
    infer = createInferRuntime(g_logger);
    assert(infer != nullptr);
    if (g_use_dla_core >= 0)
    {
        infer->setDLACore(g_use_dla_core);
    }
    engine = infer->deserializeCudaEngine(buf, size, plugin);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    std::cout<<"Create TensorRT Engine Done!"<<std::endl;

    //determine the input size and the output size by the engine, and allocate the device memory for buffers 
    assert(engine->getNbBindings() == 2);
    input_dims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(0));
    output_dims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(1));
    input_size = batch_size * input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * sizeof(float);
    output_size = batch_size * output_dims.d[0] * output_dims.d[1] * output_dims.d[2] *sizeof(float);
    std::cout<<"Input Dim: "<<batch_size<<"x"<<input_dims.d[0]<<"x"<<input_dims.d[1]<<"x"<<input_dims.d[2]<<std::endl;
    std::cout<<"Output Dim: "<<batch_size<<"x"<<output_dims.d[0]<<"x"<<output_dims.d[1]<<"x"<<output_dims.d[2]<<std::endl;
    INPUT_W = input_dims.d[2];
    INPUT_H = input_dims.d[1];
    OUTPUT_W = output_dims.d[2];
    OUTPUT_H = output_dims.d[1];
    cls_num = output_dims.d[0];

    //set the blocks and threads per block for the transfer from prob map to the rgb image on an GPU
    blocks = dim3((OUTPUT_W + threads_per_block - 1) / threads_per_block, (OUTPUT_H  + threads_per_block - 1) / threads_per_block);
    threads = dim3(threads_per_block, threads_per_block);

    //set the size of the outputed segmentation image 
    real_out.create(OUTPUT_W, OUTPUT_H, CV_8UC3);

    CHECK(cudaMalloc(&buffers[0], input_size ));
    CHECK(cudaMalloc(&buffers[1], output_size ));
    CHECK( cudaMalloc(&rgb, 3 * OUTPUT_W * OUTPUT_H * sizeof(uchar)) );
    CHECK(cudaMalloc(&gpu_img_buf, img_w * img_h * 3 * sizeof(uchar)));
    CHECK(cudaMalloc(&gpu_img_resize_buf, INPUT_W * INPUT_H * 3 * sizeof(uchar)));
    CHECK(cudaMalloc(&gpu_data_buf, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CHECK(cudaMalloc(&gpu_data_planes, INPUT_W * INPUT_H * 3 * sizeof(float)));

    CHECK(cudaStreamCreate(&stream));
    
    free(buf);
    delete plugin;
}

DeepLab::~DeepLab()
{
    context->destroy();
    engine->destroy();
    infer->destroy();
    
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaFree(rgb));
    
    cudaStreamDestroy(stream);
    CHECK(cudaFree(gpu_img_buf));
    CHECK(cudaFree(gpu_img_resize_buf));
    CHECK(cudaFree(gpu_data_buf));
    CHECK(cudaFree(gpu_data_planes));
}

//preprocess the image using CUDA npp lib
void DeepLab::make_input(const cv::Mat &img){
    Npp32f m_scale[3] = {0.00392157, 0.00392157, 0.00392157};
    //Npp32f a_scale[3] = {-1, -1, -1};
    Npp32f* r_plane = (Npp32f*)(this->gpu_data_planes);
    Npp32f* g_plane = (Npp32f*)(this->gpu_data_planes + this->INPUT_W*this->INPUT_H*sizeof(float) );
    Npp32f* b_plane = (Npp32f*)(this->gpu_data_planes + this->INPUT_W*this->INPUT_H*2*sizeof(float) );
    Npp32f* dst_planes[3] = {r_plane, g_plane, b_plane};
    int aDstOrder[3] = {2, 1, 0};
    uchar* img_data = img.data;
    int width_in = img.cols;
    int height_in = img.rows;
    NppiSize srcSize = {width_in, height_in};
    NppiSize dstSize = {this->INPUT_W, this->INPUT_H};
    NppiRect srcROI = {0, 0, width_in, height_in};
    NppiRect dstROI = {0, 0, this->INPUT_W, this->INPUT_H};
    cudaMemcpy(this->gpu_img_buf, img_data, width_in*height_in*3, cudaMemcpyHostToDevice);
    nppiResize_8u_C3R((Npp8u*)gpu_img_buf, width_in*3, srcSize, srcROI, (Npp8u*)this->gpu_img_resize_buf, this->INPUT_W*3, dstSize, dstROI,NPPI_INTER_LINEAR);
    nppiSwapChannels_8u_C3IR((Npp8u*)this->gpu_img_resize_buf, this->INPUT_W*3, dstSize, aDstOrder);
    nppiConvert_8u32f_C3R((Npp8u*)this->gpu_img_resize_buf, this->INPUT_W*3, (Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dstSize);
    nppiMulC_32f_C3IR(m_scale, (Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dstSize);
    //nppiAddC_32f_C3IR(a_scale, (Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dstSize);
    nppiCopy_32f_C3P3R((Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dst_planes, this->INPUT_W*sizeof(float), dstSize);
    //cudaMemcpy(input_data, gpu_data_planes, INPUT_W*INPUT_H*3*sizeof(float), cudaMemcpyDeviceToHost);
}

void DeepLab::doInference(const cv::Mat &img)
{
#ifdef TIME //set the timer
    cudaEvent_t start, end;
    //CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreate(&start));
    //CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start, stream));
#endif

    make_input(img);
    CHECK(cudaMemcpyAsync(buffers[0], gpu_data_planes, input_size, cudaMemcpyDeviceToDevice, stream));
    context->enqueue(batch_size, buffers, stream, nullptr);
    prob2rgb();
    cudaStreamSynchronize(stream);
    CHECK( cudaMemcpy(real_out.data, rgb, sizeof(uchar) * 3 * OUTPUT_W * OUTPUT_H, cudaMemcpyDeviceToHost) );

#ifdef TIME //record the inference time
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    float ms{0.0f};
    CHECK(cudaEventElapsedTime(&ms, start, end));
    std::cout << "Deeplab inferance time: " << ms << std::endl;  
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
#endif
}
void DeepLab::prob2rgb(){
	prob2rgb_kernel<<<blocks, threads, 0, stream>>>((float *)buffers[1], (uchar *)rgb, OUTPUT_W * OUTPUT_H);
}

int DeepLab::get_OUTPUT_W() const{
    return OUTPUT_W;
}
int DeepLab::get_OUTPUT_H() const{
    return OUTPUT_H;
}
