#include "FBP.h"

int main(int argc, char** argv )
{

    Mat src;
    src = imread( argv[1], 0);
    Mat filtered_sinogram;

    //projection and filtered
    projection_filtered_timing(src, filtered_sinogram);
    imwrite("filtered_sinogram.png", filtered_sinogram);

    //back projection
    Mat reconstruction(filtered_sinogram.size().height,filtered_sinogram.size().height,CV_64F);

    int height = src.rows;
    int weight = src.cols;
    double diagonal = sqrt(height * height + weight * weight);
    int num_of_angle = 180;
    int num_of_projection = diagonal;
    int angle_interval = 180/num_of_angle;
 
    Timer tt;
    tt.tic();
    backprojection(reconstruction, filtered_sinogram, num_of_projection, num_of_angle);
    normalization(reconstruction);
    printf("Openmp backprojection time: %6.4f\n", tt.toc());

    imwrite("reconstructed.png", reconstruction);
    return 0;
}