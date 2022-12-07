/*
 * Coding Project #3: Vector Addition in Parrallel, GPU version.
 * Emanuel Francis
 * CSC 656 
*/

#include <iostream>
#include <math.h>
#include <chrono>

// CUDA Kernel function to add the elements of two arrays on the GPU

__global__
void add(int n, float *x, float *y)
{
    // n arithmetic operations
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] + y[i];
    }
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
    add<<<1, 1>>>(N, x, y);

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
