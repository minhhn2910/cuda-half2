#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 threadIdx;
THREE_VECTOR d;
int first_j;
double fs;
double fxij;
double fyij;
double fzij;
int pointer;
double r2;
double u2;
double vij;
double fA_shared[(4*100)];
double qB_shared[100];
double rA_shared[(4*100)];
double rB_shared[(4*100)];
if (enableSignal_compute)
{
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
k=0;
}
while (k<(1+d_box_gpu_nn[blockIdx.x]))
{
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
if ((k==0))
{
pointer=blockIdx.x;
}
else
{
pointer=d_box_gpu_number[(((blockIdx.x*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[threadIdx.x]=d_rv_gpu[((4*first_j)+threadIdx.x)];
qB_shared[threadIdx.x]=d_qv_gpu[(first_j+threadIdx.x)];
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
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
}
}

#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
{
int enableSignal_compute;
dim3 blockIdx;
double a2;
int first_i;
int k;
int j;
blockIdx.y=0;
blockIdx.x=core_id;
while (1)
{
while (blockIdx.x>=gridDim.x)
{
blockIdx.x=(blockIdx.x-gridDim.x);
blockIdx.y=(blockIdx.y+1);
}
if ((blockIdx.y>=gridDim.y))
{
break;
}
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
}
kernel_gpu_cuda_compute((enableSignal_compute&&(blockIdx.x<number_boxes)), blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
blockIdx.x=(blockIdx.x+num_cores);
}
}

