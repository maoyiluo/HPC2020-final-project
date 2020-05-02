#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>    // std::max
#include <fstream>
#include "utils.h"
using namespace cv;

__global__ 
void iterate(Mat &reconstruction, Mat filtered_sinogram, int num_of_angle, int num_of_projection){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double pixel = 0;
    if(row < num_of_projection && col < num_of_projection){
        for(int t=0; t<num_of_angle; t++)
        {
            int rho= (row-0.5*num_of_projection + 0.5)*cos(delta_t*t) - (col + 0.5 -0.5*num_of_projection)*sin(delta_t*t) + 0.5*num_of_projection;
            if((rho>=0)&&(rho<num_of_projection)) pixel += filtered_sinogram.at<double>(rho,t);
        }
        if(pixel<0) pixel=0;
        reconstruction.at<double>(row,col)= pixel;
    }
}
