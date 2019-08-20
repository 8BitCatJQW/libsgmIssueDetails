/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#include"../include/common.h"

using namespace std;

namespace StereoMatching
{

   

    cv::Mat stereoMatching(cv::Mat I1, cv::Mat I2)
    {
        const int disp_size = 128;

        ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
        ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
        ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
        ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

        const int width = I1.cols;
        const int height = I1.rows;

        const int input_depth = I1.type() == CV_8U ? 8 : 16;
        const int input_bytes = input_depth * width * height / 8;
        const int output_depth = 8;
        const int output_bytes = output_depth * width * height / 8;

        static sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

        cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);
        cv::Mat disparity_8u, disparity_color;

        device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);



        const auto t1 = std::chrono::system_clock::now();
        cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

        cout << d_I1.data << endl;
        sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
        cudaDeviceSynchronize();

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
        cout << duration/1000 << endl;

        return disparity;
    }
    

    void StereoMatchingSGM::initializerSGM(const int& width, const int& height, const int& disp_size, const int& imgType)
    {
        this->width = width;
        this->height = height;
        this->disp_size = disp_size;
        ASSERT_MSG(imgType == CV_8U , "input image format must be CV_8U or CV_16U.");
        input_depth = imgType == CV_8U ? 8 : 16;
        input_bytes = input_depth * width * height / 8;
        output_depth = 8;
        output_bytes = output_depth * width * height / 8;
        sgm::StereoSGM  stereoSGM1(width , height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
        stereoSGM = &stereoSGM1;

       // stereoSGM.setStereoSGM(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
      
    
    }

    cv::Mat StereoMatchingSGM::executeSGM(cv::Mat I1, cv::Mat I2)
    {
        ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
        ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
        ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
        ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

        cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);

        device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
        cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

        stereoSGM->execute(d_I1.data, d_I2.data, d_disparity.data);
        cudaDeviceSynchronize();
        cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

        return disparity;
    }


    sgm::StereoSGM initializerSGM(const int& width, const int& height, const int& disp_size, const int& imgType)
    {
        
        ASSERT_MSG(imgType == CV_8U, "input image format must be CV_8U or CV_16U.");
        int input_depth = imgType == CV_8U ? 8 : 16;
        int input_bytes = input_depth * width * height / 8;
        int output_depth = 8;
        int output_bytes = output_depth * width * height / 8;

        sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

    
    }

 
}
