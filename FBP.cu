#include "utils.h"
#include "FBP.h"
#include <omp.h>


__global__
void filtered_kernel(double *sinogram_device, double* filtered_sinogram, double* filter,int num_of_projection, int num_of_angle){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0;
    int half_num_projection = num_of_projection/2;
    if(row < num_of_projection && col < num_of_projection){
        for (int k = 0; k < num_of_projection; k++)
        {
            if(half_num_projection + row - k >=0)
            local_sum += sinogram_device[k * num_of_angle + col] * filter[half_num_projection + row - k];
        }
        filtered_sinogram[row * num_of_angle + col] = local_sum;
    }
}

__global__ 
void backprojection_kernel(double *reconstruction, double* filtered_sinogram, int num_of_angle, int num_of_projection){
    float delta_t;
    delta_t=1.0*M_PI/num_of_angle;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double pixel = 0;
    if(row < num_of_projection && col < num_of_projection){
        for(int t=0; t<num_of_angle; t++)
        {
            int rho= (row-0.5*num_of_projection + 0.5)*cos(delta_t*t) - (col + 0.5 -0.5*num_of_projection)*sin(delta_t*t) + 0.5*num_of_projection;
            if((rho>=0)&&(rho<num_of_projection)) pixel += filtered_sinogram[rho * num_of_projection + t];
        }
        if(pixel<0) pixel=0;
        reconstruction[row * num_of_projection + col]= pixel;
    }
}

int main(int argc, char** argv )
{
    Mat src;
    src = imread( argv[1], 0);
    Mat filtered_sinogram;

    //projection and filtered
    int height = src.rows;
    int weight = src.cols;

    double diagonal = sqrt(height * height + weight * weight);
    int padding_top = (diagonal - height) / 2;
    int padding_bottom = (diagonal - height) / 2;
    int padding_left = (diagonal - weight) / 2;
    int padding_right = (diagonal - weight) / 2;
    Mat padding_src;
    copyMakeBorder(src, padding_src, padding_top, padding_bottom, padding_left, padding_right, BORDER_CONSTANT, 0);
    Point2f center((padding_src.cols - 1) / 2.0, (padding_src.rows - 1) / 2.0);

    int max_pixel = 0;
    for (int i = 0; i < padding_src.rows; i++)
    {
        for (int j = 0; j < padding_src.cols; j++)
        {
            if (padding_src.at<uchar>(i, j) > max_pixel)
                max_pixel = padding_src.at<uchar>(i, j);
        }
    }

    int num_of_angle = 180;
    int num_of_projection = diagonal;
    int angle_interval = 180 / num_of_angle;

    Mat sinogram = Mat::zeros(num_of_projection, num_of_angle, CV_64F);

    Timer tt;
    //projection
    tt.tic();
    projection(padding_src, sinogram, num_of_angle, center, angle_interval);
    imwrite("sinogram.png", sinogram);
    printf("Openmp Projection time: %6.4f\n", tt.toc());

    //opencv convolution filtered
    tt.tic();
    filter(sinogram, filtered_sinogram, num_of_projection);
    printf("opencv filtered time: %6.4f\n", tt.toc());

    //openmp convolution
    tt.tic();
    my_filter(sinogram, filtered_sinogram, num_of_projection);
    printf("Openmp filtered time: %6.4f\n", tt.toc());

    //cuda convolution
    Mat filter = Mat::zeros(num_of_projection, 1, CV_64F);
    int half_num_projection = num_of_projection / 2;
    #pragma omp parallel for
    for (int i = 1; i < half_num_projection; i = i + 2)
    {
        filter.at<double>(half_num_projection - i, 1) = -1.0 / (i * i * M_PI * M_PI);
        if (half_num_projection + i < num_of_projection)
            filter.at<double>(half_num_projection + i, 1) = -1.0 / (i * i * M_PI * M_PI);
    }
    double* sinogram_device;
    double* filtered_sinogram_device;
    double* filtered_device;
    tt.tic();
    double omp_time = omp_get_wtime();
    cudaMalloc<double>(&sinogram_device, num_of_projection*num_of_angle);
    cudaMalloc<double>(&filtered_sinogram_device, num_of_projection*num_of_angle);
    cudaMalloc<double>(&filtered_device, num_of_projection);
    cudaMemcpy(sinogram_device,sinogram.ptr(),num_of_projection * num_of_projection,cudaMemcpyHostToDevice);
    cudaMemcpy(filtered_device,filter.ptr(),num_of_projection,cudaMemcpyHostToDevice);
    dim3 filter_block(256, 256);
    dim3 filter_grid((num_of_angle + 256- 1)/256, (num_of_projection  + 256 - 1)/256);
    filtered_kernel<<<filter_grid, filter_block>>>(sinogram_device, filtered_sinogram_device, filtered_device, num_of_projection, num_of_angle);
    cudaDeviceSynchronize();
    cudaMemcpy(filter.ptr(),filtered_device, num_of_projection*sizeof(double), cudaMemcpyDeviceToHost);
    printf("Cuda filtered time: %6.4f\n", tt.toc());
    printf("Cuda omp_get_wtime filtered time: %6.4f\n", omp_get_wtime() - omp_time);

    //back projection
    Mat reconstruction(filtered_sinogram.size().height,filtered_sinogram.size().height,CV_64F);
  
    tt.tic();
    backprojection(reconstruction, filtered_sinogram, num_of_projection, num_of_angle);
    printf("Openmp backprojection time: %6.4f\n", tt.toc());

    double* reconstruction_device;

    tt.tic();
    omp_time = omp_get_wtime();
    cudaMalloc<double>(&reconstruction_device, num_of_projection * num_of_projection);
    
    cudaMemcpy(sinogram_device,filtered_sinogram.ptr(),num_of_projection * num_of_projection,cudaMemcpyHostToDevice);
    dim3 block(256,256);
    dim3 grid((num_of_projection + block.x - 1)/block.x, (num_of_projection  + block.y - 1)/block.y);

    backprojection_kernel<<<grid, block>>>(reconstruction_device, sinogram_device, num_of_angle, num_of_projection);
    cudaDeviceSynchronize();
    cudaMemcpy(reconstruction.ptr(),reconstruction_device, num_of_projection * num_of_projection*sizeof(double), cudaMemcpyDeviceToHost);
    printf("cuda backprojection time: %6.4f\n", tt.toc());
    printf("Cuda omp_get_wtime filtered time: %6.4f\n", omp_get_wtime() - omp_time);
    normalization(reconstruction);

    imwrite("reconstructed_cuda.png", reconstruction);
    return 0;
}