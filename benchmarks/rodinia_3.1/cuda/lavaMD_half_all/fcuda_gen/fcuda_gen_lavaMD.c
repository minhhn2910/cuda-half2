//~ #include <fcuda.h>
//~ #include "main.h"
#include "/home/minh/github/fcuda/fcuda_src/include/fcuda.h"
#include "/home/minh/github/fcuda-benchmarks/lavaMD/main.h"
#include <string.h>
#include <math.h>
#include <mpfr.h>
#include <stdio.h>
#define LEN 15 
 int config_vals[LEN];
        mpfr_t temp_var_1;
        mpfr_t temp_var_2;
        mpfr_t temp_var_3;
        mpfr_t temp_var_4;
        mpfr_t temp_var_5;
        mpfr_t temp_var_6;
        mpfr_t temp_var_7;
        mpfr_t temp_var_8;
        mpfr_t temp_var_9;
        mpfr_t temp_var_10;
        mpfr_t temp_var_11;
        mpfr_t temp_var_12;
        mpfr_t temp_var_13;
        mpfr_t temp_var_14;
        mpfr_t temp_var_15;
        mpfr_t temp_var_16;
        mpfr_t temp_var_17;
        mpfr_t temp_var_18;
  mpfr_t fs_kernel_gpu_cuda_compute;
  mpfr_t fxij_kernel_gpu_cuda_compute;
  mpfr_t fyij_kernel_gpu_cuda_compute;
  mpfr_t fzij_kernel_gpu_cuda_compute;
  mpfr_t r2_kernel_gpu_cuda_compute;
  mpfr_t u2_kernel_gpu_cuda_compute;
  mpfr_t vij_kernel_gpu_cuda_compute;
  mpfr_t a2_temp_kernel_gpu_cuda_compute;
        mpfr_t temp_var_19;
        mpfr_t temp_var_20;
        mpfr_t temp_var_21;
        mpfr_t temp_var_22;
        mpfr_t temp_var_23;
        mpfr_t temp_var_24;
        mpfr_t temp_var_25;
        mpfr_t temp_var_26;
        mpfr_t temp_var_27;
        mpfr_t temp_var_28;
        mpfr_t temp_var_29;
        mpfr_t temp_var_30;
  mpfr_t fA_shared_temp_kernel_gpu_cuda_compute;
  mpfr_t qB_shared_temp_kernel_gpu_cuda_compute;
  mpfr_t rA_shared_temp_kernel_gpu_cuda_compute;
  mpfr_t rB_shared_temp_kernel_gpu_cuda_compute;
                    mpfr_t temp_var_31;
                    mpfr_t temp_var_32;
                    mpfr_t temp_var_33;
                    mpfr_t temp_var_34;
                    mpfr_t temp_var_35;
                    mpfr_t temp_var_36;
                    mpfr_t temp_var_37;
                    mpfr_t temp_var_38;
                    mpfr_t temp_var_39;
                    mpfr_t temp_var_40;
  mpfr_t a2_block0_kernel_gpu_cuda;
  mpfr_t alpha_temp_kernel_gpu_cuda;
    mpfr_t temp_var_41;
            mpfr_t temp_var_42;
            mpfr_t temp_var_43;
            mpfr_t temp_var_44;
            mpfr_t temp_var_45;
            mpfr_t temp_var_46;
            mpfr_t temp_var_47;
                mpfr_t temp_var_48;
                mpfr_t temp_var_49;
                mpfr_t temp_var_50;
                mpfr_t temp_var_51;
                mpfr_t temp_var_52;
            mpfr_t temp_var_53;
        mpfr_t temp_var_54;
int init_mpfr() { 
        mpfr_init2 (temp_var_1, config_vals[0]);
        mpfr_init2 (temp_var_2, config_vals[0]);
        mpfr_init2 (temp_var_3, config_vals[0]);
        mpfr_init2 (temp_var_4, config_vals[0]);
        mpfr_init2 (temp_var_5, config_vals[0]);
        mpfr_init2 (temp_var_6, config_vals[0]);
        mpfr_init2 (temp_var_7, config_vals[0]);
        mpfr_init2 (temp_var_8, config_vals[0]);
        mpfr_init2 (temp_var_9, config_vals[0]);
        mpfr_init2 (temp_var_10, config_vals[0]);
        mpfr_init2 (temp_var_11, config_vals[0]);
        mpfr_init2 (temp_var_12, config_vals[0]);
        mpfr_init2 (temp_var_13, config_vals[0]);
        mpfr_init2 (temp_var_14, config_vals[0]);
        mpfr_init2 (temp_var_15, config_vals[0]);
        mpfr_init2 (temp_var_16, config_vals[0]);
        mpfr_init2 (temp_var_17, config_vals[0]);
        mpfr_init2 (temp_var_18, config_vals[0]);
  mpfr_init2(fs_kernel_gpu_cuda_compute, config_vals[1]);
  mpfr_init2(fxij_kernel_gpu_cuda_compute, config_vals[2]);
  mpfr_init2(fyij_kernel_gpu_cuda_compute, config_vals[3]);
  mpfr_init2(fzij_kernel_gpu_cuda_compute, config_vals[4]);
  mpfr_init2(r2_kernel_gpu_cuda_compute, config_vals[5]);
  mpfr_init2(u2_kernel_gpu_cuda_compute, config_vals[6]);
  mpfr_init2(vij_kernel_gpu_cuda_compute, config_vals[7]);
  mpfr_init2(a2_temp_kernel_gpu_cuda_compute, config_vals[8]);
        mpfr_init2 (temp_var_19, config_vals[0]);
        mpfr_init2 (temp_var_20, config_vals[0]);
        mpfr_init2 (temp_var_21, config_vals[0]);
        mpfr_init2 (temp_var_22, config_vals[0]);
        mpfr_init2 (temp_var_23, config_vals[0]);
        mpfr_init2 (temp_var_24, config_vals[0]);
        mpfr_init2 (temp_var_25, config_vals[0]);
        mpfr_init2 (temp_var_26, config_vals[0]);
        mpfr_init2 (temp_var_27, config_vals[0]);
        mpfr_init2 (temp_var_28, config_vals[0]);
        mpfr_init2 (temp_var_29, config_vals[0]);
        mpfr_init2 (temp_var_30, config_vals[0]);
  mpfr_init2(fA_shared_temp_kernel_gpu_cuda_compute, config_vals[9]);
  mpfr_init2(qB_shared_temp_kernel_gpu_cuda_compute, config_vals[10]);
  mpfr_init2(rA_shared_temp_kernel_gpu_cuda_compute, config_vals[11]);
  mpfr_init2(rB_shared_temp_kernel_gpu_cuda_compute, config_vals[12]);
                    mpfr_init2 (temp_var_31, config_vals[0]);
                    mpfr_init2 (temp_var_32, config_vals[0]);
                    mpfr_init2 (temp_var_33, config_vals[0]);
                    mpfr_init2 (temp_var_34, config_vals[0]);
                    mpfr_init2 (temp_var_35, config_vals[0]);
                    mpfr_init2 (temp_var_36, config_vals[0]);
                    mpfr_init2 (temp_var_37, config_vals[0]);
                    mpfr_init2 (temp_var_38, config_vals[0]);
                    mpfr_init2 (temp_var_39, config_vals[0]);
                    mpfr_init2 (temp_var_40, config_vals[0]);
  mpfr_init2(a2_block0_kernel_gpu_cuda, config_vals[13]);
  mpfr_init2(alpha_temp_kernel_gpu_cuda, config_vals[14]);
    mpfr_init2 (temp_var_41, config_vals[0]);
            mpfr_init2 (temp_var_42, config_vals[0]);
            mpfr_init2 (temp_var_43, config_vals[0]);
            mpfr_init2 (temp_var_44, config_vals[0]);
            mpfr_init2 (temp_var_45, config_vals[0]);
            mpfr_init2 (temp_var_46, config_vals[0]);
            mpfr_init2 (temp_var_47, config_vals[0]);
                mpfr_init2 (temp_var_48, config_vals[0]);
                mpfr_init2 (temp_var_49, config_vals[0]);
                mpfr_init2 (temp_var_50, config_vals[0]);
                mpfr_init2 (temp_var_51, config_vals[0]);
                mpfr_init2 (temp_var_52, config_vals[0]);
            mpfr_init2 (temp_var_53, config_vals[0]);
        mpfr_init2 (temp_var_54, config_vals[0]);
}

int init_readconfig() {

  // For reading precision contents of config_file into array
   FILE *myFile;
     myFile = fopen("config_file.txt", "r");
 
        if (myFile == NULL) {
					printf("Error Reading File\n");
                exit (0);
                }
 
        int s;
        for (s = 0; s < LEN-1; s++) {
            fscanf(myFile, "%d,", &config_vals[s+1]);
                          }
		config_vals[0] = 53; //all temp_vars are 53 bits in mantissa
        fclose(myFile);
        init_mpfr();
        return 0;             
}

void kernel_gpu_cuda_fetch1(int enableSignal_fetch1, dim3 blockDim, dim3 gridDim, double * d_rv_gpu, double rA_shared[(4*100)], int fetch1_rA_shared_offset, int fetch1_rA_shared_X_0, int fetch1_rA_shared_c_1, double * d_fv_gpu, double fA_shared[(4*100)], int fetch1_fA_shared_offset, int fetch1_fA_shared_X_0, int fetch1_fA_shared_c_1)
{
dim3 threadIdx;
if (enableSignal_fetch1)
{
memcpy((fetch1_rA_shared_offset+rA_shared), ((d_rv_gpu+fetch1_rA_shared_X_0)+(threadIdx.y*fetch1_rA_shared_c_1)), (400*sizeof (double)));
memcpy((fA_shared+fetch1_fA_shared_offset), ((d_fv_gpu+fetch1_fA_shared_X_0)+(threadIdx.y*fetch1_fA_shared_c_1)), (400*sizeof (double)));
}
}

#pragma fcuda transfer array_split=[rA_shared] mpart=1 dir=[1] name=write pointer=[d_fv_gpu] type=burst cores=[1] end=false unroll=1 begin=true size=[400] 
void kernel_gpu_cuda_write(int enableSignal_write, dim3 blockDim, dim3 gridDim, double * d_fv_gpu, double fA_shared[(4*100)], int write_fA_shared_offset, int write_fA_shared_X_0, int write_fA_shared_c_1)
{
dim3 threadIdx;
if (enableSignal_write)
{
memcpy(((d_fv_gpu+write_fA_shared_X_0)+(threadIdx.y*write_fA_shared_c_1)), (fA_shared+write_fA_shared_offset), (400*sizeof (double)));
}
}

#pragma fcuda transfer array_split=[rA_shared] mpart=1 dir=[0|0] name=fetch2 pointer=[d_rv_gpu|d_qv_gpu] type=burst cores=[1] end=false unroll=1 begin=true size=[400|100] 
void kernel_gpu_cuda_fetch2(int enableSignal_fetch2, dim3 blockDim, dim3 gridDim, double * d_rv_gpu, double rB_shared[(4*100)], int fetch2_rB_shared_offset, int fetch2_rB_shared_X_0, int fetch2_rB_shared_c_1, double * d_qv_gpu, double qB_shared[100], int fetch2_qB_shared_offset, int fetch2_qB_shared_X_0, int fetch2_qB_shared_c_1)
{
dim3 threadIdx;
if (enableSignal_fetch2)
{
memcpy((fetch2_rB_shared_offset+rB_shared), ((d_rv_gpu+fetch2_rB_shared_X_0)+(threadIdx.y*fetch2_rB_shared_c_1)), (400*sizeof (double)));
memcpy((fetch2_qB_shared_offset+qB_shared), ((d_qv_gpu+fetch2_qB_shared_X_0)+(threadIdx.y*fetch2_qB_shared_c_1)), (100*sizeof (double)));
}
}

void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, double fA_shared[4 * 100], int j, double qB_shared[100], double rA_shared[4 * 100], double rB_shared[4 * 100])
{
  dim3 threadIdx;
  THREE_VECTOR d;
  
	mpfr_set_d(a2_temp_kernel_gpu_cuda_compute, a2, MPFR_RNDN);

  if (enableSignal_compute)
  {
    for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x = threadIdx.x + 1){
    {
      if (threadIdx.x < 100)
      {
        for (j = 0; j < (4 * 100); j += 4){
        {
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[4 * threadIdx.x], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j], MPFR_RNDN);
                              mpfr_add(r2_kernel_gpu_cuda_compute, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 1], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 1], MPFR_RNDN);
          
                    mpfr_mul(temp_var_31, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
                    mpfr_sub(r2_kernel_gpu_cuda_compute, r2_kernel_gpu_cuda_compute, (temp_var_31), MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 2], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 2], MPFR_RNDN);
          
                    mpfr_mul(temp_var_32, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
                    mpfr_sub(r2_kernel_gpu_cuda_compute, r2_kernel_gpu_cuda_compute, (temp_var_32), MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 3], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 3], MPFR_RNDN);
          
                    mpfr_mul(temp_var_33, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
                    mpfr_sub(r2_kernel_gpu_cuda_compute, r2_kernel_gpu_cuda_compute, (temp_var_33), MPFR_RNDN);
                              mpfr_mul(u2_kernel_gpu_cuda_compute, a2_temp_kernel_gpu_cuda_compute, r2_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_set_d(vij_kernel_gpu_cuda_compute, exp(-mpfr_get_d(u2_kernel_gpu_cuda_compute, MPFR_RNDN)), MPFR_RNDN);
                              mpfr_mul_si(fs_kernel_gpu_cuda_compute, vij_kernel_gpu_cuda_compute, 2, MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 1], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 1], MPFR_RNDN);
          
                    mpfr_sub(temp_var_34, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);

d.x = mpfr_get_d(temp_var_34, MPFR_RNDN);
                              mpfr_mul_d(fxij_kernel_gpu_cuda_compute, fs_kernel_gpu_cuda_compute, d.x, MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 2], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 2], MPFR_RNDN);
          
                    mpfr_sub(temp_var_35, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);

d.y = mpfr_get_d(temp_var_35, MPFR_RNDN);
                              mpfr_mul_d(fyij_kernel_gpu_cuda_compute, fs_kernel_gpu_cuda_compute, d.y, MPFR_RNDN);
          mpfr_set_d(rA_shared_temp_kernel_gpu_cuda_compute, (double) rA_shared[(4 * threadIdx.x) + 3], MPFR_RNDN);
          mpfr_set_d(rB_shared_temp_kernel_gpu_cuda_compute, (double) rB_shared[j + 3], MPFR_RNDN);
          
                    mpfr_sub(temp_var_36, rA_shared_temp_kernel_gpu_cuda_compute, rB_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);

d.z = mpfr_get_d(temp_var_36, MPFR_RNDN);
                              mpfr_mul_d(fzij_kernel_gpu_cuda_compute, fs_kernel_gpu_cuda_compute, d.z, MPFR_RNDN);
          mpfr_set_d(qB_shared_temp_kernel_gpu_cuda_compute, (double) qB_shared[j / 4], MPFR_RNDN);
          mpfr_set_d(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared[4 * threadIdx.x], MPFR_RNDN);
          
                    mpfr_mul(temp_var_37, qB_shared_temp_kernel_gpu_cuda_compute, vij_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_add(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared_temp_kernel_gpu_cuda_compute, temp_var_37, MPFR_RNDN);
          
fA_shared[4 * threadIdx.x] = mpfr_get_d(fA_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_set_d(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared[(4 * threadIdx.x) + 1], MPFR_RNDN);
          
                    mpfr_mul(temp_var_38, qB_shared_temp_kernel_gpu_cuda_compute, fxij_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_add(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared_temp_kernel_gpu_cuda_compute, temp_var_38, MPFR_RNDN);
          
fA_shared[(4 * threadIdx.x) + 1] = mpfr_get_d(fA_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_set_d(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared[(4 * threadIdx.x) + 2], MPFR_RNDN);
          
                    mpfr_mul(temp_var_39, qB_shared_temp_kernel_gpu_cuda_compute, fyij_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_add(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared_temp_kernel_gpu_cuda_compute, temp_var_39, MPFR_RNDN);
          
fA_shared[(4 * threadIdx.x) + 2] = mpfr_get_d(fA_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_set_d(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared[(4 * threadIdx.x) + 3], MPFR_RNDN);
          
                    mpfr_mul(temp_var_40, qB_shared_temp_kernel_gpu_cuda_compute, fzij_kernel_gpu_cuda_compute, MPFR_RNDN);
          mpfr_add(fA_shared_temp_kernel_gpu_cuda_compute, fA_shared_temp_kernel_gpu_cuda_compute, temp_var_40, MPFR_RNDN);
          
fA_shared[(4 * threadIdx.x) + 3] = mpfr_get_d(fA_shared_temp_kernel_gpu_cuda_compute, MPFR_RNDN);
        }

                }

      }

    }

        }

  }

}

void kernel_gpu_cuda(double alpha, long number_boxes, long *d_box_gpu_offset, int *d_box_gpu_nn, int *d_box_gpu_number, double *d_rv_gpu, double *d_qv_gpu, double *d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
{
  int enableSignal_fetch1_block0;
  int fetch1_rA_shared_offset_block0;
  int fetch1_rA_shared_X_0_block0;
  int fetch1_rA_shared_c_1_block0;
  int fetch1_fA_shared_offset_block0;
  int fetch1_fA_shared_X_0_block0;
  int fetch1_fA_shared_c_1_block0;
  int enableSignal_fetch2_block0;
  int fetch2_rB_shared_offset_block0;
  int fetch2_rB_shared_X_0_block0;
  int fetch2_rB_shared_c_1_block0;
  int fetch2_qB_shared_offset_block0;
  int fetch2_qB_shared_X_0_block0;
  int fetch2_qB_shared_c_1_block0;
  int enableSignal_compute_block0;
  int enableSignal_write_block0;
  int write_fA_shared_offset_block0;
  int write_fA_shared_X_0_block0;
  int write_fA_shared_c_1_block0;
  dim3 blockIdx_block0;
  double fA_shared_block0[4 * 100];
  double rA_shared_block0[4 * 100];
  mpfr_set_d(alpha_temp_kernel_gpu_cuda, alpha, MPFR_RNDN);
  int first_i_block0;
  int pointer_block0;
  int k_block0;
  int first_j_block0;
  int j_block0;
  int whileLoopGuard_0_block0;
  int pingpong_0;
  double rB_shared_ping_block0[4 * 100];
  double qB_shared_ping_block0[100];
  double qB_shared_pong_block0[100];
  double rB_shared_pong_block0[4 * 100];
  int whileLoopGuard_0_pipe_1_block0;
  blockIdx_block0.y = 0;
  blockIdx_block0.x = (core_id * 1) + 0;
  while (1
)
  {
    while (blockIdx_block0.x>=gridDim.x)
    {
      blockIdx_block0.x = blockIdx_block0.x - gridDim.x;
      blockIdx_block0.y = blockIdx_block0.y + 1;
    }

    if (
(blockIdx_block0.y>=gridDim.y)
)
    {
      break;
    }

    enableSignal_write_block0 = (blockIdx_block0.x < gridDim.x) && (blockIdx_block0.y < gridDim.y);
    enableSignal_compute_block0 = (blockIdx_block0.x < gridDim.x) && (blockIdx_block0.y < gridDim.y);
    enableSignal_fetch2_block0 = (blockIdx_block0.x < gridDim.x) && (blockIdx_block0.y < gridDim.y);
    enableSignal_fetch1_block0 = (blockIdx_block0.x < gridDim.x) && (blockIdx_block0.y < gridDim.y);
    if (blockIdx_block0.x < number_boxes)
    {
      
            mpfr_mul_d(temp_var_44, alpha_temp_kernel_gpu_cuda, 2.0, MPFR_RNDN);
            mpfr_mul(a2_block0_kernel_gpu_cuda, (temp_var_44), alpha_temp_kernel_gpu_cuda, MPFR_RNDN);
      k_block0 = 0;
      j_block0 = 0;
      first_i_block0 = d_box_gpu_offset[blockIdx_block0.x];
      fetch1_rA_shared_offset_block0 = 0;
      fetch1_rA_shared_X_0_block0 = 4 * first_i_block0;
      fetch1_rA_shared_c_1_block0 = 0;
      fetch1_fA_shared_offset_block0 = 0;
      fetch1_fA_shared_X_0_block0 = 4 * first_i_block0;
      fetch1_fA_shared_c_1_block0 = 0;
    }

    kernel_gpu_cuda_fetch1(enableSignal_fetch1_block0 && (blockIdx_block0.x < number_boxes), blockDim, gridDim, d_rv_gpu, rA_shared_block0, fetch1_rA_shared_offset_block0, fetch1_rA_shared_X_0_block0, fetch1_rA_shared_c_1_block0, d_fv_gpu, fA_shared_block0, fetch1_fA_shared_offset_block0, fetch1_fA_shared_X_0_block0, fetch1_fA_shared_c_1_block0);
    if (
(blockIdx_block0.x<number_boxes)
)
    {
      whileLoopGuard_0_block0 = 1;
      k_block0 = 0;
    }

    pingpong_0 = 0;
    whileLoopGuard_0_pipe_1_block0 = 0;
    while (1)
    {
      whileLoopGuard_0_block0 &= k_block0 < (1 + d_box_gpu_nn[blockIdx_block0.x]);
      if ((( ! whileLoopGuard_0_block0)&&( ! whileLoopGuard_0_pipe_1_block0))
)
      {
        break;
      }

      if (whileLoopGuard_0_block0)
      {
        if (k_block0 == 0)
        {
          pointer_block0 = blockIdx_block0.x;
        }
        else
        {
          pointer_block0 = d_box_gpu_number[((blockIdx_block0.x * 26) + k_block0) - 1];
        }

        first_j_block0 = d_box_gpu_offset[pointer_block0];
        fetch2_rB_shared_offset_block0 = 0;
        fetch2_rB_shared_X_0_block0 = 4 * first_j_block0;
        fetch2_rB_shared_c_1_block0 = 0;
        fetch2_qB_shared_offset_block0 = 0;
        fetch2_qB_shared_X_0_block0 = first_j_block0;
        fetch2_qB_shared_c_1_block0 = 0;
      }

      if ((pingpong_0==0)
)
      {
        kernel_gpu_cuda_fetch2(enableSignal_fetch2_block0 && whileLoopGuard_0_block0, blockDim, gridDim, d_rv_gpu, rB_shared_ping_block0, fetch2_rB_shared_offset_block0, fetch2_rB_shared_X_0_block0, fetch2_rB_shared_c_1_block0, d_qv_gpu, qB_shared_ping_block0, fetch2_qB_shared_offset_block0, fetch2_qB_shared_X_0_block0, fetch2_qB_shared_c_1_block0);
        kernel_gpu_cuda_compute(enableSignal_compute_block0 && whileLoopGuard_0_pipe_1_block0, blockDim, gridDim, blockIdx_block0, mpfr_get_d(a2_block0_kernel_gpu_cuda, MPFR_RNDN), fA_shared_block0, j_block0, qB_shared_pong_block0, rA_shared_block0, rB_shared_pong_block0);
        pingpong_0 = 1;
      }
      else
      {
        kernel_gpu_cuda_fetch2(enableSignal_fetch2_block0 && whileLoopGuard_0_block0, blockDim, gridDim, d_rv_gpu, rB_shared_pong_block0, fetch2_rB_shared_offset_block0, fetch2_rB_shared_X_0_block0, fetch2_rB_shared_c_1_block0, d_qv_gpu, qB_shared_pong_block0, fetch2_qB_shared_offset_block0, fetch2_qB_shared_X_0_block0, fetch2_qB_shared_c_1_block0);
        kernel_gpu_cuda_compute(enableSignal_compute_block0 && whileLoopGuard_0_pipe_1_block0, blockDim, gridDim, blockIdx_block0, mpfr_get_d(a2_block0_kernel_gpu_cuda, MPFR_RNDN), fA_shared_block0, j_block0, qB_shared_ping_block0, rA_shared_block0, rB_shared_ping_block0);
        pingpong_0 = 0;
      }

      whileLoopGuard_0_pipe_1_block0 = whileLoopGuard_0_block0;
      if (

(whileLoopGuard_0_block0)

)
      {
        k_block0++;
      }

    }

    if (blockIdx_block0.x < number_boxes)
    {
      write_fA_shared_offset_block0 = 0;
      write_fA_shared_X_0_block0 = 4 * first_i_block0;
      write_fA_shared_c_1_block0 = 0;
    }

    kernel_gpu_cuda_write(enableSignal_write_block0 && (blockIdx_block0.x < number_boxes), blockDim, gridDim, d_fv_gpu, fA_shared_block0, write_fA_shared_offset_block0, write_fA_shared_X_0_block0, write_fA_shared_c_1_block0);
    blockIdx_block0.x = blockIdx_block0.x + (num_cores * 1);
  }

}

void fcuda1(double alpha, long number_boxes, int d_box_gpu_offset_addr, int d_box_gpu_nn_addr, int d_box_gpu_number_addr, int d_rv_gpu_addr, int d_qv_gpu_addr, int d_fv_gpu_addr, dim3 gridDim, dim3 blockDim, long *memport_core0_p0, int *memport_core0_p1, double *memport_core0_p2)
{
  #pragma HLS INTERFACE ap_none register port=alpha
  #pragma HLS RESOURCE core=AXI4LiteS variable=alpha
  #pragma HLS INTERFACE ap_none register port=number_boxes
  #pragma HLS RESOURCE core=AXI4LiteS variable=number_boxes
  #pragma HLS INTERFACE ap_none register port=d_box_gpu_offset_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_box_gpu_offset_addr
  #pragma HLS INTERFACE ap_none register port=d_box_gpu_nn_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_box_gpu_nn_addr
  #pragma HLS INTERFACE ap_none register port=d_box_gpu_number_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_box_gpu_number_addr
  #pragma HLS INTERFACE ap_none register port=d_rv_gpu_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_rv_gpu_addr
  #pragma HLS INTERFACE ap_none register port=d_qv_gpu_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_qv_gpu_addr
  #pragma HLS INTERFACE ap_none register port=d_fv_gpu_addr
  #pragma HLS RESOURCE core=AXI4LiteS variable=d_fv_gpu_addr
  #pragma HLS INTERFACE ap_none register port=gridDim
  #pragma HLS RESOURCE core=AXI4LiteS variable=gridDim
  #pragma HLS INTERFACE ap_none register port=blockDim
  #pragma HLS RESOURCE core=AXI4LiteS variable=blockDim
  #pragma HLS RESOURCE core=AXI4LiteS variable=return
  #pragma HLS interface ap_bus port=memport_core0_p0
  #pragma HLS RESOURCE variable=memport_core0_p0 core=AXI4M
  #pragma HLS interface ap_bus port=memport_core0_p1
  #pragma HLS RESOURCE variable=memport_core0_p1 core=AXI4M
  #pragma HLS interface ap_bus port=memport_core0_p2
  #pragma HLS RESOURCE variable=memport_core0_p2 core=AXI4M
  int *d_box_gpu_nn_core0;
  int *d_box_gpu_number_core0;
  double *d_rv_gpu_core0;
  double *d_qv_gpu_core0;
  double *d_fv_gpu_core0;
  long *d_box_gpu_offset_core0;
  d_fv_gpu_core0 = &memport_core0_p2[d_fv_gpu_addr];
  d_qv_gpu_core0 = &memport_core0_p2[d_qv_gpu_addr];
  d_rv_gpu_core0 = &memport_core0_p2[d_rv_gpu_addr];
  d_box_gpu_number_core0 = &memport_core0_p1[d_box_gpu_number_addr];
  d_box_gpu_nn_core0 = &memport_core0_p1[d_box_gpu_nn_addr];
  d_box_gpu_offset_core0 = &memport_core0_p0[d_box_gpu_offset_addr];
  kernel_gpu_cuda(alpha, number_boxes, d_box_gpu_offset_core0, d_box_gpu_nn_core0, d_box_gpu_number_core0, d_rv_gpu_core0, d_qv_gpu_core0, d_fv_gpu_core0, gridDim, blockDim, 1, 0);
}

void fcuda(double alpha, long number_boxes, int d_box_gpu_offset_addr, int d_box_gpu_nn_addr, int d_box_gpu_number_addr, int d_rv_gpu_addr, int d_qv_gpu_addr, int d_fv_gpu_addr, dim3 gridDim, dim3 blockDim, long *memport_core0_p0_long, int *memport_core0_p1_int, double *memport_core0_p2_double, int en_fcuda1)
{
  #pragma HLS INTERFACE ap_none register port=en_fcuda1
  #pragma HLS RESOURCE core=AXI4LiteS variable=en_fcuda1
  #pragma HLS RESOURCE core=AXI4LiteS variable=return
  if (
(en_fcuda1==1)
)
  {
    fcuda1(alpha, number_boxes, d_box_gpu_offset_addr, d_box_gpu_nn_addr, d_box_gpu_number_addr, d_rv_gpu_addr, d_qv_gpu_addr, d_fv_gpu_addr, gridDim, blockDim, memport_core0_p0_long, memport_core0_p1_int, memport_core0_p2_double);
  }

}

//end of conversion, hopefully it will work :)
