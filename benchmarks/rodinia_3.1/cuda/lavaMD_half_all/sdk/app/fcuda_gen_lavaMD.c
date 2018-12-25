#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda transfer array_split=[rA_shared] mpart=1 dir=[0|0] name=fetch1 pointer=[d_rv_gpu|d_fv_gpu] type=burst cores=[1] end=false unroll=1 begin=true size=[400|400] 
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

#pragma fcuda compute array_split=[rA_shared] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, double fA_shared[(4*100)], int j, double qB_shared[100], double rA_shared[(4*100)], double rB_shared[(4*100)])
{
dim3 threadIdx;
THREE_VECTOR d;
double fs;
double fxij;
double fyij;
double fzij;
double r2;
double u2;
double vij;
if (enableSignal_compute)
{
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
if ((threadIdx.x<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*threadIdx.x)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*threadIdx.x)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*threadIdx.x)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*threadIdx.x)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*threadIdx.x)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*threadIdx.x)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*threadIdx.x)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*threadIdx.x)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*threadIdx.x)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*threadIdx.x)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*threadIdx.x)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
}
}
}

#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
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
double rA_shared_block0[(4*100)];
double fA_shared_block0[(4*100)];
double a2_block0;
int first_i_block0;
int pointer_block0;
int k_block0;
int first_j_block0;
int j_block0;
int whileLoopGuard_0_block0;
int pingpong_0;
double rB_shared_ping_block0[(4*100)];
double qB_shared_ping_block0[100];
double qB_shared_pong_block0[100];
double rB_shared_pong_block0[(4*100)];
int whileLoopGuard_0_pipe_1_block0;
blockIdx_block0.y=0;
blockIdx_block0.x=((core_id*1)+0);
while (1)
{
while (blockIdx_block0.x>=gridDim.x)
{
blockIdx_block0.x=(blockIdx_block0.x-gridDim.x);
blockIdx_block0.y=(blockIdx_block0.y+1);
}
if ((blockIdx_block0.y>=gridDim.y))
{
break;
}
enableSignal_write_block0=((blockIdx_block0.x<gridDim.x)&&(blockIdx_block0.y<gridDim.y));
enableSignal_compute_block0=((blockIdx_block0.x<gridDim.x)&&(blockIdx_block0.y<gridDim.y));
enableSignal_fetch2_block0=((blockIdx_block0.x<gridDim.x)&&(blockIdx_block0.y<gridDim.y));
enableSignal_fetch1_block0=((blockIdx_block0.x<gridDim.x)&&(blockIdx_block0.y<gridDim.y));
if ((blockIdx_block0.x<number_boxes))
{
a2_block0=((2.0*alpha)*alpha);
k_block0=0;
j_block0=0;
first_i_block0=d_box_gpu_offset[blockIdx_block0.x];
fetch1_rA_shared_offset_block0=0;
fetch1_rA_shared_X_0_block0=(4*first_i_block0);
fetch1_rA_shared_c_1_block0=0;
fetch1_fA_shared_offset_block0=0;
fetch1_fA_shared_X_0_block0=(4*first_i_block0);
fetch1_fA_shared_c_1_block0=0;
}
kernel_gpu_cuda_fetch1((enableSignal_fetch1_block0&&(blockIdx_block0.x<number_boxes)), blockDim, gridDim, d_rv_gpu, rA_shared_block0, fetch1_rA_shared_offset_block0, fetch1_rA_shared_X_0_block0, fetch1_rA_shared_c_1_block0, d_fv_gpu, fA_shared_block0, fetch1_fA_shared_offset_block0, fetch1_fA_shared_X_0_block0, fetch1_fA_shared_c_1_block0);
if ((blockIdx_block0.x<number_boxes))
{
whileLoopGuard_0_block0=1;
k_block0=0;
}
pingpong_0=0;
whileLoopGuard_0_pipe_1_block0=0;
while (1)
{
whileLoopGuard_0_block0&=k_block0<(1+d_box_gpu_nn[blockIdx_block0.x]);
if ((( ! whileLoopGuard_0_block0)&&( ! whileLoopGuard_0_pipe_1_block0)))
{
break;
}
if (whileLoopGuard_0_block0)
{
if ((k_block0==0))
{
pointer_block0=blockIdx_block0.x;
}
else
{
pointer_block0=d_box_gpu_number[(((blockIdx_block0.x*26)+k_block0)-1)];
}
first_j_block0=d_box_gpu_offset[pointer_block0];
fetch2_rB_shared_offset_block0=0;
fetch2_rB_shared_X_0_block0=(4*first_j_block0);
fetch2_rB_shared_c_1_block0=0;
fetch2_qB_shared_offset_block0=0;
fetch2_qB_shared_X_0_block0=first_j_block0;
fetch2_qB_shared_c_1_block0=0;
}
if ((pingpong_0==0))
{
kernel_gpu_cuda_fetch2((enableSignal_fetch2_block0&&whileLoopGuard_0_block0), blockDim, gridDim, d_rv_gpu, rB_shared_ping_block0, fetch2_rB_shared_offset_block0, fetch2_rB_shared_X_0_block0, fetch2_rB_shared_c_1_block0, d_qv_gpu, qB_shared_ping_block0, fetch2_qB_shared_offset_block0, fetch2_qB_shared_X_0_block0, fetch2_qB_shared_c_1_block0);
kernel_gpu_cuda_compute((enableSignal_compute_block0&&whileLoopGuard_0_pipe_1_block0), blockDim, gridDim, blockIdx_block0, a2_block0, fA_shared_block0, j_block0, qB_shared_pong_block0, rA_shared_block0, rB_shared_pong_block0);
pingpong_0=1;
}
else
{
kernel_gpu_cuda_fetch2((enableSignal_fetch2_block0&&whileLoopGuard_0_block0), blockDim, gridDim, d_rv_gpu, rB_shared_pong_block0, fetch2_rB_shared_offset_block0, fetch2_rB_shared_X_0_block0, fetch2_rB_shared_c_1_block0, d_qv_gpu, qB_shared_pong_block0, fetch2_qB_shared_offset_block0, fetch2_qB_shared_X_0_block0, fetch2_qB_shared_c_1_block0);
kernel_gpu_cuda_compute((enableSignal_compute_block0&&whileLoopGuard_0_pipe_1_block0), blockDim, gridDim, blockIdx_block0, a2_block0, fA_shared_block0, j_block0, qB_shared_ping_block0, rA_shared_block0, rB_shared_ping_block0);
pingpong_0=0;
}
whileLoopGuard_0_pipe_1_block0=whileLoopGuard_0_block0;
if (whileLoopGuard_0_block0)
{
k_block0 ++ ;
}
}
if ((blockIdx_block0.x<number_boxes))
{
write_fA_shared_offset_block0=0;
write_fA_shared_X_0_block0=(4*first_i_block0);
write_fA_shared_c_1_block0=0;
}
kernel_gpu_cuda_write((enableSignal_write_block0&&(blockIdx_block0.x<number_boxes)), blockDim, gridDim, d_fv_gpu, fA_shared_block0, write_fA_shared_offset_block0, write_fA_shared_X_0_block0, write_fA_shared_c_1_block0);
blockIdx_block0.x=(blockIdx_block0.x+(num_cores*1));
}
}

#pragma fcuda portmerge port_id=0 offset=d_box_gpu_offset_addr remove_port_name=d_box_gpu_offset_core0 port_core=0 
#pragma fcuda portmerge port_id=1 offset=d_box_gpu_nn_addr remove_port_name=d_box_gpu_nn_core0 port_core=0 
#pragma fcuda portmerge port_id=1 offset=d_box_gpu_number_addr remove_port_name=d_box_gpu_number_core0 port_core=0 
#pragma fcuda portmerge port_id=2 offset=d_rv_gpu_addr remove_port_name=d_rv_gpu_core0 port_core=0 
#pragma fcuda portmerge port_id=2 offset=d_qv_gpu_addr remove_port_name=d_qv_gpu_core0 port_core=0 
#pragma fcuda portmerge port_id=2 offset=d_fv_gpu_addr remove_port_name=d_fv_gpu_core0 port_core=0 
void fcuda1(double alpha, long number_boxes, int d_box_gpu_offset_addr, int d_box_gpu_nn_addr, int d_box_gpu_number_addr, int d_rv_gpu_addr, int d_qv_gpu_addr, int d_fv_gpu_addr, dim3 gridDim, dim3 blockDim, long * memport_core0_p0, int * memport_core0_p1, double * memport_core0_p2)
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
int * d_box_gpu_nn_core0;
int * d_box_gpu_number_core0;
double * d_rv_gpu_core0;
double * d_qv_gpu_core0;
double * d_fv_gpu_core0;
long * d_box_gpu_offset_core0;
d_fv_gpu_core0=( & memport_core0_p2[d_fv_gpu_addr]);
d_qv_gpu_core0=( & memport_core0_p2[d_qv_gpu_addr]);
d_rv_gpu_core0=( & memport_core0_p2[d_rv_gpu_addr]);
d_box_gpu_number_core0=( & memport_core0_p1[d_box_gpu_number_addr]);
d_box_gpu_nn_core0=( & memport_core0_p1[d_box_gpu_nn_addr]);
d_box_gpu_offset_core0=( & memport_core0_p0[d_box_gpu_offset_addr]);
kernel_gpu_cuda(alpha, number_boxes, d_box_gpu_offset_core0, d_box_gpu_nn_core0, d_box_gpu_number_core0, d_rv_gpu_core0, d_qv_gpu_core0, d_fv_gpu_core0, gridDim, blockDim, 1, 0);
}

void fcuda(double alpha, long number_boxes, int d_box_gpu_offset_addr, int d_box_gpu_nn_addr, int d_box_gpu_number_addr, int d_rv_gpu_addr, int d_qv_gpu_addr, int d_fv_gpu_addr, dim3 gridDim, dim3 blockDim, long * memport_core0_p0_long, int * memport_core0_p1_int, double * memport_core0_p2_double, int en_fcuda1)
{
#pragma HLS INTERFACE ap_none register port=en_fcuda1 
#pragma HLS RESOURCE core=AXI4LiteS variable=en_fcuda1 
#pragma HLS RESOURCE core=AXI4LiteS variable=return 
if ((en_fcuda1==1))
{
fcuda1(alpha, number_boxes, d_box_gpu_offset_addr, d_box_gpu_nn_addr, d_box_gpu_number_addr, d_rv_gpu_addr, d_qv_gpu_addr, d_fv_gpu_addr, gridDim, blockDim, memport_core0_p0_long, memport_core0_p1_int, memport_core0_p2_double);
}
}

