/*
 * Coding Project #3: Vector Addition in Parrallel, GPU version.
 * Emanuel Francis
 * CSC 656 
*/

#include <iostream>
#include <math.h>
#include <chrono>


__global__
void add(int n, float *x, float *y)
{
//  3 + n arithmetic operations
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
    // 3n bytes written/read

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 24;

    float *x ;
    float *y ;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initalize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // Run kernel on 1M elements on the GPU
    /*
        Together, the blocks of parallel threads make up what is known as 
        the grid. Since I have N elements to process, and 256 threads per 
        block, I just need to calculate the number of blocks to get at 
        least N threads. I simply divide N by the block size (being 
        careful to round up in case N is not a multiple of blockSize).
    */
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();


    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error " << maxError << std::endl;

    // Free Memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
