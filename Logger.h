#ifndef LOGGER_H_
#define LOGGER_H_

#include "NvInfer.h"
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger
{
    Severity reportableSeverity;
public:
    Logger(Severity severity = Severity::kWARNING);
    void log(Severity severity, const char* msg) override;
};
#endif
