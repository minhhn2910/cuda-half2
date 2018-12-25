

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C"
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  //~ float out_err, hid_err;
  float *out_err = eo;
  float *hid_err = eh;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;


  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;

  num_blocks = in / BLOCK_SIZE;
  dim3  grid( 1 , num_blocks);
  dim3  threads(BLOCK_SIZE , BLOCK_SIZE);

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }

  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));


  printf("Performing GPU computation\n");

  printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

	float time_kernel = 0.0;
    float tmp_t;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);

  cudaThreadSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	time_kernel += tmp_t;


  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }


 bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);



  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);


	cudaEventRecord(start,0);

  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,
												hid,
												input_cuda,
												in,
												input_hidden_cuda,
												input_prev_weights_cuda
												);



	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	time_kernel += tmp_t;

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);


/*
  float max_rel_err_input = 0.0;
  float max_rel_err_weights = 0.0;
  int max_err_index;
  float rel_err= 0.0;
  FILE *pFile;
  pFile = fopen("gold_output.txt", "r");
  if (pFile == NULL) {
    printf ("err open file gold_output\n");
  }
  //fprintf(pFile, "net->input_units\n");
  float gold_input_units_val;
  for (int k = 0; k < in + 1; k++) {
    //fprintf(pFile, "%f\n", net->input_units[k]);
    fscanf(pFile, "%f\n", &gold_input_units_val);
    if(net->input_units[k] != 0){
      rel_err = fabs((gold_input_units_val - net->input_units[k])/net->input_units[k]);
    }
    else
      rel_err = fabs((gold_input_units_val));
    if (rel_err > max_rel_err_input){
      max_rel_err_input = rel_err;
      max_err_index = k;
    }
  }

  float gold_weights_one_dim_val;
  //fprintf(pFile, "input_weights_one_dim\n");
  for (int k = 0; k < (in + 1) * (hid + 1); k++) {
    //fprintf(pFile, "%f\n", input_weights_one_dim[k]);
    fscanf(pFile, "%f\n", &gold_weights_one_dim_val);
    if (input_weights_one_dim[k]!=0)
        rel_err = fabs((gold_weights_one_dim_val - input_weights_one_dim[k])/input_weights_one_dim[k]);
    else
        rel_err = fabs(gold_weights_one_dim_val);
    if(rel_err > max_rel_err_weights)
      max_rel_err_weights = rel_err;

  }

  printf ("%f, %f \n",max_rel_err_input,max_rel_err_weights);

*/

 FILE *pFile;
 pFile = fopen("gold_output.txt", "w");
 if (pFile == NULL) {
 fputs("fopen example", pFile);
   return;
 }
 //fprintf(pFile, "net->input_units\n");
 /*for (int k = 0; k < in + 1; k++)
 fprintf(pFile, "%f\n", net->input_units[k]);
 */
 //fprintf(pFile, "input_weights_one_dim\n");
 for (int k = 0; k < (in + 1) * (hid + 1); k++)
 fprintf(pFile, "%f, ", input_weights_one_dim[k]);

printf ("time %f \n ", time_kernel);

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);



}
