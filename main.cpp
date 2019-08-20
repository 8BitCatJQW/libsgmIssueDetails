#include "GxIAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include<thread>
#include<chrono>
#include<memory>
#include <mutex>
#include <queue>


#include "include/CommonProcessingUnit.h"
#include "include/common.h"


using namespace pcl;
using namespace std;
using namespace cv;

using namespace CommonProcessingUnit;
using namespace StereoMatching;



bool working = true;




int main()
{


    //-----------------------SGM Initialization-------------------------------
   
    int disp_size = 128;

  
    
    Mat I1 = cv::imread("../image/left.png",0);
    Mat I2=  cv::imread("../image/right.png",0);
    std::cout << I1.cols << I2.rows << std::endl;
 
  
    ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
    ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
    ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
    ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

    int width = I1.cols;
    int height = I1.rows;
    cout << I1.type() << endl;

    const int input_depth = I1.type() == CV_8U ? 8 : 16;
    const int input_bytes = input_depth * width * height / 8;
    const int output_depth = 16;
    const int output_bytes = output_depth * width * height / 8;

    sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
    device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
    cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);

    
   
    while (1)
    {
      
        
           
                Mat stereoImg;
                hconcat(I1, I2, stereoImg);
                cv::namedWindow("STEREO", CV_WINDOW_NORMAL);
                imshow("STEREO", stereoImg);

             
              
                cv::Mat leftR = I1.clone();
                cv::Mat rightR = I2.clone();
                cudaMemcpy(d_I1.data, leftR.data, input_bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_I2.data, rightR.data, input_bytes, cudaMemcpyHostToDevice);
                sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
                cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
                if (disparity.empty())
                {
                    cout << "data empty" << endl;
                    break;
                }
                Mat result;
                cv::FileStorage fs1("../image/disp2.yml", cv::FileStorage::WRITE);
                fs1 << "disparity" << disparity;
                fs1.release();

               
                drawColorDisparity(disparity,  128);
                                    
               

            int flag = cv::waitKey(1);
            if (flag == 27)
            {
                working = false;
                break;
            }

            if (flag == 32)
            {
                //cv::Mat disp8;
               // disparity.convertTo(disp8, CV_8UC1, 1.0 / 16.);
                cv::imwrite("../image/d.png", disparity);
                testTo3D(disparity);
            }

     }

    

   

    return 0;
}

