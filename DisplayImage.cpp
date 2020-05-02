#include "utils.h"
#include "FBP.h"

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

    Timer tt;
    //projection
    tt.tic();
    projection(padding_src, sinogram, num_of_angle, center, angle_interval);
    imwrite("sinogram.png", sinogram);
    printf("Openmp Projection time: %6.4f\n", tt.toc());

    //filtered
    Mat filtered_sinogram;
    filter(sinogram, filtered_sinogram, num_of_projection);

    imwrite("filtered_sinogram.png", filtered_sinogram);

    //back projection
    Mat reconstruction(filtered_sinogram.size().height,filtered_sinogram.size().height,CV_64F);
 
    tt.tic();
    backprojection(reconstruction, filtered_sinogram, num_of_projection, num_of_angle);
    normalization(reconstruction);
    printf("Openmp backprojection time: %6.4f\n", tt.toc());

    imwrite("reconstructed.png", reconstruction);
    return 0;
}