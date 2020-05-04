#include "utils.h"
#include "FBP.h"


__global__
void filtered_kernel(double *sinogram_device, double* filtered_sinogram, double* filter,int num_of_projection, int num_of_angle){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0;
    if(row < num_of_projection && col < num_of_projection){
        for (int k = 0; k < num_of_projection; k++)
        {
            if(half_num_projection + row - k >=0)
                sum += sinogram[k * num_of_angle + col] * filter[half_num_projection + row - k];
        }
        filtered_sinogram[row * num_of_angle + col] = sum;
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
    projection_filtered_timing(src, filtered_sinogram);
    imwrite("filtered_sinogram.png", filtered_sinogram);

    int height = src.rows;
    int weight = src.cols;
    double diagonal = sqrt(height * height + weight * weight);
    int num_of_angle = 180;
    int num_of_projection = diagonal;
    int angle_interval = 180/num_of_angle;
 
    //back projection
    Mat reconstruction(filtered_sinogram.size().height,filtered_sinogram.size().height,CV_64F);
  
    Timer tt;
    tt.tic();
    backprojection(reconstruction, filtered_sinogram, num_of_projection, num_of_angle);
    printf("Openmp backprojection time: %6.4f\n", tt.toc());

    double* reconstruction_device;
    double* sinogram_device;
    tt.tic();
    cudaMalloc<double>(&reconstruction_device, num_of_projection * num_of_projection);
    cudaMalloc<double>(&sinogram_device, num_of_projection*num_of_angle);

    cudaMemcpy(sinogram_device,filtered_sinogram.ptr(),num_of_projection * num_of_projection,cudaMemcpyHostToDevice);
    dim3 block(256,256);
    dim3 grid((num_of_projection + block.x - 1)/block.x, (num_of_projection  + block.y - 1)/block.y);

    backprojection_kernel<<<grid, block>>>(reconstruction_device, sinogram_device, num_of_angle, num_of_projection);
    cudaDeviceSynchronize();
    cudaMemcpy(reconstruction.ptr(),reconstruction_device, num_of_projection * num_of_projection*sizeof(double), cudaMemcpyDeviceToHost);
    printf("cuda backprojection time: %6.4f\n", tt.toc());
    normalization(reconstruction);

    imwrite("reconstructed_cuda.png", reconstruction);
    return 0;
}