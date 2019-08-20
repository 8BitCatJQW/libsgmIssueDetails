#ifndef __STEREO_MATCHING_H__
#define __STEREO_MATCHING_H__

#include "GxIAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <memory>
#include <chrono>
#include <ctime>
#include <windows.h>
#include <iostream>
#include<string>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui_c.h>


#include <cuda_runtime.h>
#include <libsgm.h>

namespace StereoMatching
{
    #define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

    struct device_buffer
    {
        device_buffer() : data(nullptr) {}
        device_buffer(size_t count) { allocate(count); }
        void allocate(size_t count) { cudaMalloc(&data, count); }
        ~device_buffer() { cudaFree(data); }
        void* data;
    };

    template <class... Args>
    static std::string format_string(const char* fmt, Args... args)
    {
        const int BUF_SIZE = 1024;

        char buf[BUF_SIZE];
        std::snprintf(buf, BUF_SIZE, fmt, args...);
        return std::string(buf);
    }

    cv::Mat stereoMatching(cv::Mat I1, cv::Mat I2);
    sgm::StereoSGM initializerSGM(const int& width, const int& height, const int& disp_size, const int& imgType);
    cv::Mat executeSGM(cv::Mat I1, cv::Mat I2);


    struct StereoMatchingSGM
    {
        int input_depth;
        int input_bytes;
        int output_depth ;
        int output_bytes;
        int width;
        int height;
        int imgType;
        int disp_size;
        sgm::StereoSGM* stereoSGM;

        StereoMatchingSGM() {};

        void initializerSGM(const int& width, const int& height, const int& disp_size, const int& imgType);
        cv::Mat executeSGM(cv::Mat I1, cv::Mat I2);
        

    };
}

#endif
