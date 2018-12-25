

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"
#include <cuda_fp16.h>
#include "half2_operator_overload.cuh"

__global__ void
bpnn_layerforward_CUDA(__half2 *input_cuda,
	                   __half2 *output_hidden_cuda,
					   __half2 *input_hidden_cuda,
					   __half2 *hidden_partial_sum,
					   int in,
					   int hid)
{
   int by = blockIdx.y;
   int tx = threadIdx.x;//div-ed by 2 
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

   int index_in = HEIGHT * by + ty + 1;

   __shared__ __half2 input_node[HEIGHT];
   __shared__ __half2 weight_matrix[HEIGHT][WIDTH/2];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in/2] ;

   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index/2];

   __syncthreads();

   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();

   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){ //reduction in y dimension, must group them in SIMD X-DIM

	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }

   //__syncthreads();

   input_hidden_cuda[index/2] = weight_matrix[ty][tx];

/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){

	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid/2 + ty] = weight_matrix[tx][ty];
   }

}


__global__ void bpnn_adjust_weights_cuda(__half2 * delta,
										 int hid,
										 __half2 * ly,
										 int in,
										 __half2 * w,
										 __half2 * oldw)
{


   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif
