#ifndef CNN_H
#define CNN_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include<opencv2/core/eigen.hpp>

using namespace cv;
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
void conv2D(Mat &src,
            Mat &dst,
            Mat kernel,
            int dst_width,
            int dst_height,
            double biase = 0,
            bool overlapping = true,
            bool rot = false);

#endif