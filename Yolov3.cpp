#include "Yolov3.h"
#include <numeric>
Yolov3::Yolov3(const int img_w, const int img_h, const float *anchors, const int mask[][3], 
                const float obj_thresh, const float nms_thresh, const char *trt_file, const int cls_num, 
                const char *cls_name_file, const int batch_size, const int g_use_dla_core)
                :batch_size(batch_size), g_use_dla_core(g_use_dla_core), 
                obj_thresh(obj_thresh), nms_thresh(nms_thresh)
{
    //get all class names
    get_class_names(cls_name_file);
    
    //read the size of trt file and read the trt file
    uchar *buf; 
    FILE* fp = fopen(trt_file, "rb");
    fseek(fp, 0L, SEEK_END); 
    int size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    buf = (unsigned char*)malloc(size);
    fread((void*)buf, 1, size, fp);
    fclose(fp);
    
    //generate the engine and the context for inference
    runtime = createInferRuntime(g_logger);
    assert(runtime != nullptr);
    if (g_use_dla_core >= 0)
    {
        runtime->setDLACore(g_use_dla_core);
    }
    IPluginFactory* plugin = nvonnxparser::createPluginFactory(g_logger);
    assert(plugin != nullptr);
    engine = runtime->deserializeCudaEngine((void*)buf, size, plugin);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    free(buf);
    delete plugin;

    //determine the input size and the output size by the engine, and allocate the device memory for buffers 
    assert(engine->getNbBindings() == 4);
    int inputIndex;
    int outputIndex;
    Dims in_dims, out_dims;
    for (int b = 0; b < engine->getNbBindings(); ++b)
    {
        if (engine->bindingIsInput(b)){
            inputIndex = b;
            in_dims = engine->getBindingDimensions(inputIndex);
            printf("in%d dims %d: %dx%dx%d\n", inputIndex, in_dims.nbDims, in_dims.d[0], in_dims.d[1], in_dims.d[2]);
            INPUT_W = in_dims.d[2];
            INPUT_H = in_dims.d[1];
            CHECK(cudaMalloc(&buffers[inputIndex], batch_size * in_dims.d[0] * in_dims.d[1] * in_dims.d[2] * sizeof(float)));
        }else{
            outputIndex = b;
            out_dims = engine->getBindingDimensions(outputIndex);
            printf("out%d dims: %dx%dx%d\n", outputIndex, out_dims.d[0], out_dims.d[1], out_dims.d[2]);
            buf_size.push_back(out_dims.d[0] * out_dims.d[1] * out_dims.d[2]);
            CHECK(cudaMalloc(&buffers[outputIndex], batch_size * buf_size.back() * sizeof(float)));
        }
    }
    OUTPUT_SIZE = batch_size * accumulate(buf_size.begin(), buf_size.end(), 0);
    result = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    result_step1 = buf_size[0];
    result_step2 = buf_size[0] + buf_size[1];
    
    //initialize three parsers for three outputs with different sizes respectively
    parser1.init(32, INPUT_W, INPUT_H, cls_num, 3, anchors, mask[0], obj_thresh);
    parser2.init(16, INPUT_W, INPUT_H, cls_num, 3, anchors, mask[1], obj_thresh);
    parser3.init(8, INPUT_W, INPUT_H, cls_num, 3, anchors, mask[2], obj_thresh);

    CHECK(cudaMalloc(&gpu_img_buf, img_w * img_h * 3 * sizeof(uchar)));
    CHECK(cudaMalloc(&gpu_img_resize_buf, INPUT_W * INPUT_H * 3 * sizeof(uchar)));
    CHECK(cudaMalloc(&gpu_data_buf, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CHECK(cudaMalloc(&gpu_data_planes, INPUT_W * INPUT_H * 3 * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
}

Yolov3::~Yolov3(){
    free(result);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaStreamDestroy(stream);
    for(int i=0;i<4;i++){
        CHECK(cudaFree(buffers[i]));
    }
    CHECK(cudaFree(gpu_img_buf));
    CHECK(cudaFree(gpu_img_resize_buf));
    CHECK(cudaFree(gpu_data_buf));
    CHECK(cudaFree(gpu_data_planes));
}

void Yolov3::doInference(const cv::Mat &img, vector<box>& box_list, vector<box>& box_result){
#ifdef TIME //set the timer
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start, stream));
#endif

    make_input(img);
    CHECK(cudaMemcpyAsync(buffers[0], gpu_data_planes, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToDevice, this->stream));
    context->enqueue(batch_size, buffers, stream, nullptr);
    float* pout = this->result;
    for(int i=0;i<3;i++)
    {
        CHECK(cudaMemcpyAsync(pout, buffers[i+1], batch_size * buf_size[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
        pout += buf_size[i];
    }
    cudaStreamSynchronize(stream);
    box_list.clear();
    box_result.clear();
    make_yolov3_result(box_list);
    nms(box_list, box_result);

#ifdef TIME  //record the inference time
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    float ms{0.0f};
    CHECK(cudaEventElapsedTime(&ms, start, end));
    std::cout << "Yolov3 inferance time: " << ms << std::endl;  
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
#endif

}


//preprocess the image using CUDA npp lib
void Yolov3::make_input(const cv::Mat &img){
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

//postprocess the bbox results and paint them on the image 
void Yolov3::make_output(cv::Mat &img, vector<box> &box_list) const{
    int len = box_list.size();
    int r,g,b,cls;
    float x1,x2,y1,y2;
    box box1;
    
    for(int i=0; i<len; i++)
    {
        box1 = box_list[i];
        cls = box1.cls + i + 1;
        
        r = (cls*1234567)%255;
        g = (cls*7654321)%255;
        b = (cls*3456787)%255;
        x1 = (box1.x - box1.w / 2) * img.cols;
        x2 = (box1.x + box1.w / 2) * img.cols;
        y1 = (box1.y - box1.h / 2) * img.rows;
        y2 = (box1.y + box1.h / 2) * img.rows;
        x1 = x1 > 0 ? x1 : 0;
        y1 = y1 > 0 ? y1 : 0;
        cv::rectangle(img, cvPoint(x1,y1),cvPoint(x2,y2),cv::Scalar(r,g,b),1,8,0);
        //cv::putText(img, cls_names[cls - 1 - i], cvPoint(x1,y1), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(r, g, b), 2, 1, 0);
        cv::putText(img, to_string(i), cvPoint(x1,y1), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(r, g, b), 2, 1, 0);
    }
}

void Yolov3::get_class_names(const char * const file){
    ifstream infile;
    infile.open(file);
    string s;
    while(getline(infile,s))
    {
        cls_names.push_back(s);
    }
    infile.close();
}

void Yolov3::print_boxes(vector<box>& box_list) const{
    int len = box_list.size();
    std::cout<<"box number: "<<len<<std::endl;
    if(len > 0)
    {
        for(int i=0; i<len; i++)
        {
            printf("%2d: %.4f, %.4f, %.4f, %.4f   prob: %.2f\n", box_list[i].cls, box_list[i].x,
                box_list[i].y, box_list[i].w, box_list[i].h, box_list[i].prob);
        }
    }
}
int Yolov3::nms_box_compare(const void* a, const void* b) {
    box* box1 = (box*)a;
    box* box2 = (box*)b;
    float diff = box1->prob - box2->prob;
    if(diff>0){
        return -1;
    }else if(diff<0){
        return 1;
    }else{
        return 0;
    }
}
void Yolov3::sort_boxlist(vector<box>& box_list) const{
    qsort(&box_list[0], box_list.size(), sizeof(box), nms_box_compare);
}

void Yolov3::make_yolov3_result(vector<box>& box_list){
    parser1.get_preds(result);
    parser2.get_preds(result + result_step1);
    parser3.get_preds(result + result_step2);

    parser1.get_boxes();
    parser2.get_boxes();
    parser3.get_boxes();

    box_list.insert(box_list.end(), parser1.box_list.begin(), parser1.box_list.end());
    box_list.insert(box_list.end(), parser2.box_list.begin(), parser2.box_list.end());
    box_list.insert(box_list.end(), parser3.box_list.begin(), parser3.box_list.end());
}
int Yolov3::nms(vector<box>& box_list, vector<box>& box_result) const{
    int i,j;
    int len = box_list.size();
    float iou;
    int* mask = (int*)calloc(len, sizeof(int));
    if(len == 0){
        return -1;
    }
    sort_boxlist(box_list);
    for(i=0; i<(len-1); i++)
    {
        if(mask[i] < 0){
            continue;
        }else{
            box_result.push_back(box_list[i]);
        }
        for(j=(i+1); j<len; j++)
        {
            if(mask[j] < 0)
            {
                continue;
            }else{
                iou = box_iou(box_list[i], box_list[j]);
                if(iou > nms_thresh){
                    mask[j] = -1;
                }
            }
        }
    }
    return 0;
}

