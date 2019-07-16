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

#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace std;
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

static const int INPUT_H = 416;
static const int INPUT_W = 416;
float data[INPUT_H * INPUT_W * 3];
static const int OUTPUT_SIZE = (85*3)*(19*19 + 38*38 + 76*76);
float result[OUTPUT_SIZE];
static Logger gLogger;
static int gUseDLACore = -1;

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

inline void enableDLA(IBuilder* b, int useDLACore)
{
    if (useDLACore >= 0)
    {
        b->allowGPUFallback(true);
        b->setFp16Mode(true);
        b->setDefaultDeviceType(DeviceType::kDLA);
        b->setDLACore(useDLACore);
    }
}

int get_stream_from_file(const char* filename, unsigned char* buf, size_t* size)
{
    FILE* fp = fopen(filename, "rb");
    if(fp == NULL)
    {
        printf("Can not open trt file\n");
        return -1;
    }else{
        fseek(fp,0L,SEEK_END); 
        *size = ftell(fp);
        fseek(fp,0L,SEEK_SET);
        int ret = fread(buf, 1, *size, fp);
        fclose(fp);
        return ret;
    }
}

void onnxToTRTModel(const char* modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory*& trtModelStream, // output buffer for the TensorRT model
                    char* out_trtfile) 
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger);

    const int stream_size = 1000;//1000MB
    unsigned char* stream_buf;
    size_t size;
    stream_buf = (unsigned char*)calloc(stream_size*1024*1024, 1);
    if(stream_buf){
        get_stream_from_file(modelFile, stream_buf, &size);
    }else{
        printf("malloc stream space failed!\n");
        return ;
    }

    //if (!parser->parseFromFile(modelFile, verbosity))
    if (!parser->parse(stream_buf, size))
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);

    enableDLA(builder, gUseDLACore);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    FILE* fp = fopen(out_trtfile, "wb");
    fwrite(trtModelStream->data(), 1, trtModelStream->size(), fp);
    fclose(fp);

    engine->destroy();
    network->destroy();
    builder->destroy();
    free(stream_buf);
}

int main(int argc, char *argv[])
{
    IHostMemory* trtModelStream{nullptr};
    onnxToTRTModel(argv[1], 1, trtModelStream, argv[2]);
    assert(trtModelStream != nullptr);

    return 0;
}
