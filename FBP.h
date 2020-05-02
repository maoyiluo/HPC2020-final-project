#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>    // std::max
#include <fstream>
using namespace cv;


void projection(Mat padding_src, Mat &sinogram, int num_of_angle, Point2f center, int angle_interval){
    #pragma omp parallel for
    for(int i = 0; i < num_of_angle; i++){
        double angle = angle_interval * i;
        Mat rot = getRotationMatrix2D(center, 90-angle, 1.0);
        Mat dst;
        warpAffine(padding_src, dst, rot, padding_src.size());
        double sum_of_col = 0;
        for(int col = 0; col < dst.cols; col++){
            sum_of_col = 0;
            for(int row = 0; row < dst.rows; row++){
                sum_of_col += dst.at<uchar>(row,col);
            }
            sinogram.at<double>(col, i) = sum_of_col;
        }
    }
}

void filter(Mat sinogram, Mat &filtered_sinogram, int num_of_projection){
    Mat filter = Mat::zeros(num_of_projection, 1, CV_64F);
    int half_num_projection = num_of_projection/2;
    for(int i = 1; i < half_num_projection; i=i+2){
        filter.at<double>(half_num_projection-i, 1) = -1.0/(i*i*M_PI*M_PI);
        if(half_num_projection + i < num_of_projection)
            filter.at<double>(half_num_projection+i, 1) = -1.0/(i*i*M_PI*M_PI);
    }
    filter.at<double>(half_num_projection, 1) = 1.0/4;
    filter2D(sinogram, filtered_sinogram, sinogram.depth(), filter);
}

void backprojection(Mat &reconstruction, Mat filtered_sinogram, int num_of_projection, int num_of_angle){
    float delta_t;
    delta_t=1.0*M_PI/filtered_sinogram.size().width;
    // unsigned int t,f,c,rho;
    double max_entry = 0;
    double min_entry = 0;

    #pragma omp parallel for
    for(int f=0;f<reconstruction.size().height;f++)
    {
        for(int c=0;c<reconstruction.size().width;c++)
        {
            reconstruction.at<double>(f,c)=0;
            for(int t=0;t<filtered_sinogram.size().width;t++)
            {
                int rho= (f-0.5*num_of_projection + 0.5)*cos(delta_t*t) - (c + 0.5 -0.5*num_of_projection)*sin(delta_t*t) + 0.5*num_of_projection;
                if((rho>=0)&&(rho<num_of_projection)) reconstruction.at<double>(f,c) += filtered_sinogram.at<double>(rho,t);
            }
            if(reconstruction.at<double>(f,c)<0) reconstruction.at<double>(f,c)=0;
        }
    }
    
    //rotate(reconstruction,reconstruction,ROTATE_90_CLOCKWISE);
}

void normalization(Mat &graph){
    double max = 0;
    int row, col;
    for(row = 0; row < graph.rows; row++){
        for(col = 0; col < graph.cols; col++){
            max = std::max(max, graph.at<double>(row,col));
        }
    }

    for(row = 0; row < graph.rows; row++){
        for(col = 0; col < graph.cols; col++){
            graph.at<double>(row,col) = graph.at<double>(row,col)/max * 255;
        }
    }
}