#include <iostream>
// #include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
////////////////////////////////////////////////////////////////////////////////
using arg_t = float;
// using Mat = cv::Mat_<arg_t>;
// using cv::Mat;
using namespace std;
using namespace cv;
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
    double data[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    Mat m = Mat(3, 3, CV_64FC1, &data);
    cout << m << endl;
    cout << repeat(m,2,2) << endl;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////