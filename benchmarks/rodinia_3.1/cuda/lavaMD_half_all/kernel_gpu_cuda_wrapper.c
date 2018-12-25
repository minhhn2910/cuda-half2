#include "main.h"                		// (in the main program folder) needed to recognized input parameters
#include "string.h"
#include "util/device/device.h"			// (in library path specified to compiler) needed by for device functions
#include "util/timer/timer.h"			// (in library path specified to compiler) needed by timer
#include "kernel_gpu_cuda_wrapper.h"		// (in the current directory)

void kernel_gpu_cuda_wrapper(par_str par_cpu,
    dim_str dim_cpu,
    box_str* box_cpu,
    fp* rv_cpu,
    fp* qv_cpu,
    fp* fv_cpu)
{
  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();
  box_str* d_box_gpu;
  fp* d_rv_gpu;
  fp* d_qv_gpu;
  fp* d_fv_gpu;

  dim3 threads;
  dim3 blocks;

  blocks.x = dim_cpu.number_boxes;
  blocks.y = 1;
  blocks.y = 1;
  threads.x = NUMBER_THREADS;	// define the number of threads in the block
  threads.y = 1;
  threads.z = 1;
  time1 = get_time();
  d_box_gpu = (box_str*)malloc(dim_cpu.box_mem);

  d_rv_gpu = (fp*)malloc(dim_cpu.space_mem);

  d_qv_gpu = (fp*)malloc(dim_cpu.space_mem2);

  d_fv_gpu = (fp*)malloc(dim_cpu.space_mem);
  time2 = get_time();

  memcpy(d_box_gpu, box_cpu, dim_cpu.box_mem);

  memcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem);

   memcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2);

   memcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem);
  time3 = get_time();

  long *d_box_gpu_offset = (long*)malloc(dim_cpu.number_boxes * sizeof(long));
  int *d_box_gpu_nn = (int*)malloc(dim_cpu.number_boxes * sizeof(int));
  int *d_box_gpu_number = (int*)malloc(dim_cpu.number_boxes * 26 * sizeof(int));
  int i, j;
  for (i = 0; i < dim_cpu.number_boxes; i++) {
    d_box_gpu_offset[i] = d_box_gpu[i].offset;
    d_box_gpu_nn[i] = d_box_gpu[i].nn;
    for (j = 0; j < 26; j++)
      d_box_gpu_number[i * 26 + j] = d_box_gpu[i].nei[j].number;
  }
  // launch kernel - all boxes
   kernel_gpu_cuda(par_cpu.alpha, dim_cpu.number_boxes, d_box_gpu_offset, d_box_gpu_nn, d_box_gpu_number, d_rv_gpu, d_qv_gpu, d_fv_gpu, blocks, threads, 1, 0);
  
  time4 = get_time();
  memcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem);
  time5 = get_time();

  free(d_rv_gpu);
  free(d_qv_gpu);
  free(d_fv_gpu);
  free(d_box_gpu);
  free(d_box_gpu_offset);
  free(d_box_gpu_nn);
  free(d_box_gpu_number);
  time6 = get_time();
/*
  printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

  printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n", (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 		 (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",		 (float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

  printf("%15.12f s, %15.12f % : GPU: KERNEL\n",			 (float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

  printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",		 (float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 		 (float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

  printf("Total time:\n");
  printf("%.12f s\n", 						 (float) (time6-time0) / 1000000);
*/
}
