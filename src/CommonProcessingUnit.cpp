#include "../include/CommonProcessingUnit.h"
#include <pcl/io/ply_io.h>

using namespace cv;

int user_data;

namespace CommonProcessingUnit
{ 
    template <class... Args>
    static std::string format_string(const char* fmt, Args... args)
    {
        const int BUF_SIZE = 1024;

        char buf[BUF_SIZE];
        std::snprintf(buf, BUF_SIZE, fmt, args...);
        return std::string(buf);
    }

   
   

    void drawDisparity(const cv::Mat&  disparity, const double& duration, const int& disp_size)
    {

        // draw results
        cv::Mat disparity_8u, disparity_color;
        imwrite("../image/disparity.png", disparity);
 
        disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);

        cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);

        imwrite("../image/disparity_color.png", disparity_color);

        const double fps = 1e3 / duration;
        cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", duration, fps),

        cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

        cv::imshow("disparity", disparity_color);

    }


  

    void drawColorDisparity(const cv::Mat&  disparity, const int& disp_size)
    {
        // draw results
        cv::Mat disparity_8u, disparity_color;
        disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
        cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);

        cv::imshow("disparity_color", disparity_color);
    }


    void testTo3D(cv::Mat disparity)
    {
      

        cv::Mat dispfraw, dispf;
        disparity.convertTo(dispfraw, CV_32FC1, 1.0 / 16.);
        dispf = dispfraw;
       

        PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

        cv::Mat xyz;
        float Q23 = 1070.7892430584152;
        float Q32 = 12.714294995450645;
        float Q33 = 0;
        float Q03 = -650.32479095458984;
        float Q13 = -497.35033798217773;
        cv::Mat_<cv::Vec3f> XYZ(dispf.rows, dispf.cols);   // Output point cloud
        cv::Mat_<double> vec_tmp(4, 1);
        for (int y = 0; y < dispf.rows; ++y) {
            for (int x = 0; x < dispf.cols; ++x) {
                cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
                float dis = dispf.at<float_t>(y, x);
                if (dis == 0) {
                    point[2] = NAN;
                    continue;
                }
                const float pw = 1.0f / (dis * Q32 + Q33);

                point[0] = (static_cast<float>(x) + Q03) * pw;
                point[1] = (static_cast<float>(y) + Q13) * pw;
                point[2] = Q23 * pw; //z = focus / (dis * Q32 + Q33)  Zc = baseline * f / (d + doffs)
            }
        }
 
        exportToPly("../image/output.ply", XYZ);


    }


    void exportToPly(const std::string& path, cv::Mat mat)
    {
        const double max_z = 3.0;
        int numbers = 0;
        for (int y = 0; y < mat.rows; y++) {
            for (int x = 0; x < mat.cols; x++) {
                cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
                if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z || isnan(point[2]))
                    continue;
                numbers++;
            }
        }

        // Write the ply file
        std::ofstream out = writePlyHeader(path, numbers, true);
        for (int y = 0; y < mat.rows; y++) {
            for (int x = 0; x < mat.cols; x++) {
                cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
                if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z || isnan(point[2]))
                    continue;

                out << point(0) << " ";
                out << point(1) << " ";
                out << point(2) << " \n";
            }
        }
    }

    std::ofstream writePlyHeader(const std::string& fname, int count, bool pointOnly)
    {
        std::ofstream out(fname);
        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment pointcloud saved from Realsense Viewer\n";
        out << "element vertex " << count << "\n";
        out << "property float" << sizeof(float) * 8 << " x\n";
        out << "property float" << sizeof(float) * 8 << " y\n";
        out << "property float" << sizeof(float) * 8 << " z\n";

        if (!pointOnly)
        {
            out << "property float" << sizeof(float) * 8 << " nx\n";
            out << "property float" << sizeof(float) * 8 << " ny\n";
            out << "property float" << sizeof(float) * 8 << " nz\n";

            out << "property uchar red\n";
            out << "property uchar green\n";
            out << "property uchar blue\n";
        }
        out << "end_header\n";

        return out;
    }





}