#include <fcuda.h>
#ifdef __cplusplus
extern "C" {
#endif
  void kernel_gpu_cuda_wrapper(par_str parms_cpu,
      dim_str dim_cpu,
      box_str* box_cpu,
      fp* rv_cpu,
      fp* qv_cpu,
      fp* fv_cpu);

  void kernel_gpu_cuda(double alpha, long number_boxes, int * d_box_gpu_offset,
      int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu,
      double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim,
      int num_cores, int core_id);

#ifdef __cplusplus
}
#endif
