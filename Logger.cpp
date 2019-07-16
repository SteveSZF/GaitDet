#include "Logger.h"
#include <iostream>
Logger::Logger(Severity severity)
    : reportableSeverity(severity)
{
}

void Logger::log(Severity severity, const char* msg)
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

   
