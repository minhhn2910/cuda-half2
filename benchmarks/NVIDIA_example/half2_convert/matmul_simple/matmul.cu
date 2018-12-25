#define N 4000

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#include <stdio.h>
__global__ void matrixMult (float *a, float *b, float *c, int width) {
 int k = 0;
 float sum = 0.0;
 int col = threadIdx.x + blockDim.x * blockIdx.x;
 int row = threadIdx.y + blockDim.y * blockIdx.y;
 if(col < width && row < width) {
   for (k = 0; k < width; k++)
      sum += a[row * width + k] * b[k * width + col];
   c[row * width + col] = sum;
 }
}
int main() {
 //float a[N][N], b[N][N], c[N][N];
 float *dev_a, *dev_b, *dev_c;
float *a,*b,*c;
a = (float*) malloc(N*N*sizeof(float));
b = (float*) malloc(N*N*sizeof(float));
c = (float*) malloc(N*N*sizeof(float));

 // initialize matrices a and b with appropriate values
 for (int i = 0; i< N ; i++)
 for (int j=0 ; j<N ; j++)
 {
   a[i*N+j] = 1;
   b[i*N+j] = 1;
 }
 int size = N * N * sizeof(float);
 cudaMalloc((void **) &dev_a, size);
 cudaMalloc((void **) &dev_b, size);
 cudaMalloc((void **) &dev_c, size);
 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
 int NumThreads = 32;
 dim3 dimGrid(DIV_UP(N,NumThreads), DIV_UP(N,NumThreads));
 dim3 dimBlock(NumThreads, NumThreads);

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
/*
 for (int i = 0; i< N ; i++){
 for (int j=0 ; j<N ; j++)
 {
   printf("%f , ", c[i][j]);

 }
 printf ("\n");
 }
 */
printf("%f, %f \n",c[0],c[N*N-1] );

 cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}
