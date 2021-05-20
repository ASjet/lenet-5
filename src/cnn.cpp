#include "cnn.h"
////////////////////////////////////////////////////////////////////////////////
void conv2D(Mat &src,
            Mat &dst,
            Mat kernel,
            int dst_width, int dst_height,
            double biase, bool overlapping, bool rot)
{
    if(rot)
        flip(kernel, kernel, -1);

    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;
    int step_x = (overlapping)? 1 : kernel_width;
    int step_y = (overlapping)? 1 : kernel_height;
    int pad_x = (kernel_width - 1) / 2;
    int pad_y = (kernel_height - 1) / 2;

    filter2D(src, dst, -1, kernel, Point(-1,-1), biase);
    dst = dst(Rect(pad_x,pad_y,dst_width,dst_height));
}

void invPool(Mat &src, int kernel_width, int kernel_height)
{
    int src_width = src.cols;
    int src_height = src.rows;

}