#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    Mat src;
    src = imread( argv[1], 0);

    int height = src.rows;
    int weight = src.cols;

    double diagonal = sqrt(height * height + weight * weight);
    int padding_top = (diagonal-height) / 2;
    int padding_bottom = (diagonal-height) / 2;
    int padding_left = (diagonal-weight) / 2;
    int padding_right = (diagonal-weight) / 2;
    Mat padding_src;
    copyMakeBorder(src, padding_src, padding_top, padding_bottom, padding_left, padding_right, BORDER_CONSTANT, 0);
    Point2f center((padding_src.cols-1)/2.0, (padding_src.rows-1)/2.0);

    int max_pixel = 0;
    for(int i = 0; i < padding_src.rows; i++){
        for(int j = 0; j < padding_src.cols; j++){
            if(padding_src.at<uchar>(i,j) > max_pixel)
                max_pixel = padding_src.at<uchar>(i,j);
        }
    }

    int num_of_angle = 180;
    int num_of_projection = diagonal;
    int angle_interval = 180/num_of_angle;

    Mat sinogram = Mat::zeros(num_of_projection, num_of_angle, CV_64F);

    //projection
    #pragma openmp parallel for
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

    imwrite("sinogram.png", sinogram);

    //filtered
    Mat filter = Mat::zeros(num_of_projection, 1, CV_64F);
    int half_num_projection = num_of_projection/2;
    for(int i = 1; i < half_num_projection; i=i+2){
        filter.at<double>(half_num_projection-i, 1) = -1.0/(i*i*M_PI*M_PI);
        if(half_num_projection + i < num_of_projection)
            filter.at<double>(half_num_projection+i, 1) = -1.0/(i*i*M_PI*M_PI);
    }
    filter.at<double>(half_num_projection, 1) = 1.0/4;

    Mat filtered_sinogram;
    filter2D(sinogram, filtered_sinogram, sinogram.depth(), filter);

    imwrite("filtered_sinogram.png", filtered_sinogram);

    //back projection
    Mat reconstruction(filtered_sinogram.size().height,filtered_sinogram.size().height,CV_64F);
    float delta_t;
    delta_t=1.0*M_PI/sinogram.size().width;
    unsigned int t,f,c,rho;
    double max_entry = 0;
    double min_entry = 0;

    #pragma openmp parallel for
    for(f=0;f<reconstruction.size().height;f++)
    {
        for(c=0;c<reconstruction.size().width;c++)
        {
            reconstruction.at<double>(f,c)=0;
            for(t=0;t<sinogram.size().width;t++)
            {
                rho= (f-0.5*num_of_projection + 0.5)*cos(delta_t*t) - (c + 0.5 -0.5*num_of_projection)*sin(delta_t*t) + 0.5*num_of_projection;
                if((rho>=0)&&(rho<sinogram.size().height)) reconstruction.at<double>(f,c)+=filtered_sinogram.at<double>(rho,t);
            }
            if(reconstruction.at<double>(f,c)<0) reconstruction.at<double>(f,c)=0;
            if(reconstruction.at<double>(f,c)>max_entry) max_entry = reconstruction.at<double>(f,c);
            if(reconstruction.at<double>(f,c)<min_entry) min_entry = reconstruction.at<double>(f,c);
        }
    }

    printf("max entry: %f\n", max_entry);
    
    #pragma openmp parallel for
    for(f=0;f<reconstruction.size().height;f++)
    {
        for(c=0;c<reconstruction.size().width;c++)
        {
            reconstruction.at<double>(f,c) = reconstruction.at<double>(f,c)/(max_entry-min_entry) * 255;
        }
    }
    //rotate(reconstruction,reconstruction,ROTATE_90_CLOCKWISE);

    imwrite("reconstructed.png", reconstruction);
    return 0;
}