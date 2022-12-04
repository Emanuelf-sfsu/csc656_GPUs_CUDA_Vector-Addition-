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

    // insert your timer code here
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // insert your end timer code here, and print out elapsed time for this problem size
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error " << maxError << std::endl;
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time is : " << elapsed.count() << " " << std::endl;

    // Free Memory
    delete[] x;
    delete[] y;

    return 0;
}