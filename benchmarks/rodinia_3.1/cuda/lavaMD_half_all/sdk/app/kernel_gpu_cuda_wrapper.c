#include "main.h"                		// (in the main program folder) needed to recognized input parameters
#include "string.h"
#include "device/device.h"			// (in library path specified to compiler) needed by for device functions
#include "kernel_gpu_cuda_wrapper.h"		// (in the current directory)
#include <math.h>
#include "xil_cache.h"
#include "platform.h"
#include "xparameters.h"
#include "xbasic_types.h"
#include "xstatus.h"
#include "xil_io.h"
#include "xfcuda.h"
#define HW
#define SW
#define VERIFY
#define EPSILON 0.001
int compareVectorFloats(float* out, float* ref, const int size, float errThreshld)
{
  // out is the vector to compare
  // ref is the reference
  // size = number of elements in both out and ref

  double error_norm = 0;
  double ref_norm = 0;
  int i;
  for (i=0; i<size; ++i) {
    const double diff = out[i] - ref[i];
    error_norm += diff * diff;
    ref_norm += ref[i]*ref[i];
  }

  error_norm = sqrt((double)error_norm);
  ref_norm = sqrt((double)ref_norm);

  if (fabs(ref_norm) < 1e-7) {
    fprintf (stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }

  double err = error_norm / ref_norm;

  printf("error =  %.4lf\n", err);
  if (err < errThreshld) {
    printf("PASSED.\n");
    return 1;
  } else {
    printf("FAILED.\n");
    return 0;
  }
}

void kernel_gpu_cuda_wrapper(par_str par_cpu,
    dim_str dim_cpu,
    box_str* box_cpu,
    fp* rv_cpu,
    fp* qv_cpu,
    fp* fv_cpu)
{
  int i, j;

#ifdef HW
  int Status, done;
  XFcuda xcore;
  Status = XFcuda_Initialize(&xcore, 0);
  if (Status != XST_SUCCESS) {
    xil_printf("Initialization failed %d\r\n", i);
    return;
  }
#endif

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

  d_box_gpu = (box_str*)malloc(dim_cpu.box_mem);
  d_rv_gpu = (fp*)malloc(dim_cpu.space_mem);
  d_qv_gpu = (fp*)malloc(dim_cpu.space_mem2);
  d_fv_gpu = (fp*)malloc(dim_cpu.space_mem);

  memcpy(d_box_gpu, box_cpu, dim_cpu.box_mem);

  memcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem);

  memcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2);

  memcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem);

  long long int *d_box_gpu_offset = (long long int*)malloc(dim_cpu.number_boxes * sizeof(long long int));
  long *d_box_gpu_offset1 = (long*)malloc(dim_cpu.number_boxes * sizeof(long));
  int *d_box_gpu_nn = (int *)malloc(dim_cpu.number_boxes * sizeof(int));
  int *d_box_gpu_number = (int *)malloc(dim_cpu.number_boxes * 26 * sizeof(int));

  for (i = 0; i < dim_cpu.number_boxes; i++) {
    d_box_gpu_offset[i] = d_box_gpu[i].offset;
    d_box_gpu_offset1[i] = d_box_gpu[i].offset;
    d_box_gpu_nn[i] = d_box_gpu[i].nn;
    for (j = 0; j < 26; j++)
      d_box_gpu_number[i * 26 + j] = d_box_gpu[i].nei[j].number;
  }


#ifdef HW
  XFcuda_SetAlpha(&xcore, par_cpu.alpha);
  XFcuda_SetNumber_boxes(&xcore, dim_cpu.number_boxes);
  XFcuda_SetGriddim_x(&xcore, blocks.x);
  XFcuda_SetGriddim_y(&xcore, blocks.y);
  //XFcuda_SetGriddim_z(&xcore, blocks.z);
  XFcuda_SetBlockdim_x(&xcore, threads.x);
  //XFcuda_SetBlockdim_y(&xcore, threads.y);
  //XFcuda_SetBlockdim_z(&xcore, threads.z);
  XFcuda_SetD_box_gpu_offset_addr(&xcore, (u32)d_box_gpu_offset / 8);
  XFcuda_SetD_box_gpu_nn_addr(&xcore, (u32)d_box_gpu_nn / 4);
  XFcuda_SetD_box_gpu_number_addr(&xcore, (u32)d_box_gpu_number / 4);
  XFcuda_SetD_rv_gpu_addr(&xcore, (u32)d_rv_gpu / 8);
  XFcuda_SetD_qv_gpu_addr(&xcore, (u32)d_qv_gpu / 8);
  XFcuda_SetD_fv_gpu_addr(&xcore, (u32)d_fv_gpu / 8);
#endif

  // launch kernel - all boxes

#ifdef HW
  XFcuda_SetEn_fcuda1(&xcore, 1);
  Xil_DCacheDisable();
  XFcuda_Start(&xcore);
  while (!XFcuda_IsDone(&xcore));
  Xil_DCacheEnable();
#endif

#ifdef SW
  kernel_gpu_cuda(par_cpu.alpha, dim_cpu.number_boxes, d_box_gpu_offset1, d_box_gpu_nn, d_box_gpu_number, rv_cpu, qv_cpu, fv_cpu, blocks, threads, 1, 0);
#endif

#ifdef VERIFY
  for (i = 0; i < 10; i++)
    printf("index %d: sw=%lf vs hw=%lf\n", i, fv_cpu[i], d_fv_gpu[i]);

  int passed = compareVectorFloats(d_fv_gpu, fv_cpu, dim_cpu.space_elem, 1e-3f);
  if (passed) {
    for (i = 0; i < dim_cpu.space_elem; i++) {
      if (d_fv_gpu[i] != d_fv_gpu[i]) {
        printf("nan at %d\n", i);
      }
      if (fv_cpu[i] - d_fv_gpu[i] < -EPSILON ||
          fv_cpu[i] - d_fv_gpu[i] > EPSILON) {
        printf("Mismatch at %d: cpu = %f, hw = %f\n", i, fv_cpu[i], d_fv_gpu[i]);
        passed = 0;
        break;
      }
    }
  }
  if (passed)
    printf("PASSED.\n");
  else
    printf("FAILED.\n");
#endif

  free(d_rv_gpu);
  free(d_qv_gpu);
  free(d_fv_gpu);
  free(d_box_gpu);
  free(d_box_gpu_offset);
  free(d_box_gpu_offset1);
  free(d_box_gpu_nn);
  free(d_box_gpu_number);
}
