#ifndef __COMMMON_PROCESSING_UNIT_H__
#define __COMMMON_PROCESSING_UNIT_H__




#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/eigen.h>
#include <pcl/common/gaussian.h>
#include <pcl/common/transforms.h>



#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h> 
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>



using namespace std;
using namespace cv;
using namespace pcl;





namespace CommonProcessingUnit
{
    
    void drawDisparity(const cv::Mat&  disparity, const double& duration, const int& disp_size);
 
    void drawColorDisparity(const cv::Mat&  disparity, const int& disp_size);
    void testTo3D(cv::Mat disparity);
    void exportToPly(const std::string& path, cv::Mat mat);
    std::ofstream writePlyHeader(const std::string& fname, int count, bool pointOnly);

   
   


}

#endif // DEBUG