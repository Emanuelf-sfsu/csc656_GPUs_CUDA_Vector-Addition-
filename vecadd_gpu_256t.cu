/*
 * Coding Project #3: Vector Addition in Parrallel, GPU version.
 * Emanuel Francis
 * CSC 656 
*/

#include <iostream>
#include <math.h>
#include <chrono>

/*
    threadIdx.x contains the index of the current thread within its block, 
    and blockDim.x contains the number of threads in the block.
    Modify the loop to stride through the array with parallel threads.
*/
__global__
void add(int n, float *x, float *y)
{
  
  // n arithmetic operations
  int index = threadIdx.x;
  int stride = blockDim.x;
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
        If I run the code with only this change, it will do the computation once 
        per thread, rather than spreading the computation across the parallel 
        threads. To do it properly, I need to modify the kernel. CUDA C++ provides 
        keywords that let kernels get the indices of the running threads.
    */
    add<<<1, 256>>>(N, x, y);

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
