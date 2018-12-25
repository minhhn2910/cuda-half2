#define N 4000
#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#include <stdio.h>
#include <cuda_fp16.h>
#include "newhalf.hpp"
#include "half2_operator_overload.cuh"
__global__ void matrixMult (__half2 *a, __half2 *b, __half2 *c, int width) {
 int k = 0;
 half* a_half = (half*)a;
 half2 sum = __float2half2_rn(0.0);
 int col = threadIdx.x + blockDim.x * blockIdx.x;
 int row = threadIdx.y + blockDim.y * blockIdx.y;
 if(col < width/2 && row < width) {
	for (k = 0; k < width; k++){
      //__half2 a_temp = __half2half2(a_half[row * width + k]);
      //~ sum += a[row * width + k] * b[k * width + col];
      sum += __half2half2(a_half[row * width + k]) * b[k * width/2 + col];
      
      }
   c[row * width/2 + col] = sum;
 }
}
int main() {
 half_float::half *a, *b, *c;
 half2 *dev_a, *dev_b, *dev_c;
a = (half_float::half*) malloc(N*N*sizeof(half_float::half));
b = (half_float::half*) malloc(N*N*sizeof(half_float::half));
c = (half_float::half*) malloc(N*N*sizeof(half_float::half));

 // initialize matrices a and b with appropri
 // initialize matrices a and b with appropriate values
 for (int i = 0; i< N ; i++)
 for (int j=0 ; j<N ; j++)
 {
   a[i*N +j] = half_float::half(0.1);
   b[i*N +j] = half_float::half(0.1);
 }


 int size = N * N * sizeof(half);
 cudaMalloc((void **) &dev_a, size);
 cudaMalloc((void **) &dev_b, size);
 cudaMalloc((void **) &dev_c, size);
 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
 int NumThreads = 32;
 dim3 dimGrid(DIV_UP(N,NumThreads), DIV_UP(N,NumThreads));
 dim3 dimBlock(NumThreads/2, NumThreads);
 matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
 //measure performance
 cudaError_t error;
	cudaDeviceSynchronize();
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));

    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));



 int nIter = 10;

    for (int j = 0; j < nIter; j++)
    {

		matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);

	}
	

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));


    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    	printf ("msec %f\n",msecPerMatrixMul); 
    	
   //end measure performance
 
 
 
 cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
 cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
 /*for (int i = 0; i< N ; i++){
 for (int j=0 ; j<N ; j++)
 {
   printf("%f , ", float(c[i][j]));

 }
 printf ("\n");
 }
*/
}
