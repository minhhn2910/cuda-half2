[LinkSymbol] 133 updates in 0.01 seconds
[AnnotParser] begin
PreAnnotation: #pragma startinclude #include <fcuda.h>

Parent: TranslationUnit 

PreAnnotation: #pragma endinclude

Parent: TranslationUnit 

PreAnnotation: #pragma startinclude #include "main.h"

Parent: TranslationUnit 

PreAnnotation: #pragma endinclude

Parent: TranslationUnit 

PreAnnotation: #pragma startinclude #include <string.h>

Parent: TranslationUnit 

PreAnnotation: #pragma endinclude

Parent: TranslationUnit 

PreAnnotation: #pragma startinclude #include <math.h>

Parent: TranslationUnit 

PreAnnotation: #pragma endinclude

Parent: TranslationUnit 

PreAnnotation: #pragma FCUDA GRID x_dim=128

Token #
Token pragma
Token FCUDA
Token GRID
Token x_dim=128
#pragma fcuda grid x_dim=128 
attached=1

PreAnnotation: #pragma FCUDA COREINFO num_cores=1 pipeline=yes

Token #
Token pragma
Token FCUDA
Token COREINFO
Token num_cores=1
Token pipeline=yes
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_box_gpu_offset port_id=0

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_box_gpu_offset
Token port_id=0
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_box_gpu_nn port_id=1

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_box_gpu_nn
Token port_id=1
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_box_gpu_number port_id=1

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_box_gpu_number
Token port_id=1
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_rv_gpu port_id=2

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_rv_gpu
Token port_id=2
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_qv_gpu port_id=2

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_qv_gpu
Token port_id=2
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
attached=1

PreAnnotation: #pragma FCUDA PORTMERGE remove_port_name=d_fv_gpu port_id=2

Token #
Token pragma
Token FCUDA
Token PORTMERGE
Token remove_port_name=d_fv_gpu
Token port_id=2
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
attached=1

DeclarationStatement: #pragma FCUDA COMPUTE cores=[1] name=compute begin unroll=1 mpart=1 array_split=[] ;

PreAnnotation: #pragma FCUDA COMPUTE cores=[1] name=compute begin unroll=1 mpart=1 array_split=[]

Token #
Token pragma
Token FCUDA
Token COMPUTE
Token cores=[1]
Token name=compute
Token begin
Token unroll=1
Token mpart=1
Token array_split=[]
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
attached=0

Parent: DeclarationStatement 

DeclarationStatement: #pragma FCUDA COMPUTE cores=[1] name=compute end unroll=1 mpart=1 array_split=[] ;

PreAnnotation: #pragma FCUDA COMPUTE cores=[1] name=compute end unroll=1 mpart=1 array_split=[]

Token #
Token pragma
Token FCUDA
Token COMPUTE
Token cores=[1]
Token name=compute
Token end
Token unroll=1
Token mpart=1
Token array_split=[]
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
attached=0

Parent: DeclarationStatement 

[AnnotParser] end in 0.04 seconds

*** Before Any Passes  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx = blockIdx.x;
int tx = threadIdx.x;
if ((bx<number_boxes))
{
double a2 = ((2.0*alpha)*alpha);
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k = 0;
int first_j;
double * rB;
double * qB;
int j = 0;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[LinkSymbol] 133 updates in 0.00 seconds
[AnnotParser] begin
[AnnotParser] end in 0.00 seconds

*** After AnnotationParser  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx = blockIdx.x;
int tx = threadIdx.x;
if ((bx<number_boxes))
{
double a2 = ((2.0*alpha)*alpha);
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k = 0;
int first_j;
double * rB;
double * qB;
int j = 0;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[SingleDeclarator] begin
[SingleDeclarator] end in 0.00 seconds
[LinkSymbol] 133 updates in 0.00 seconds

*** After SingleDeclarator  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx = blockIdx.x;
int tx = threadIdx.x;
if ((bx<number_boxes))
{
double a2 = ((2.0*alpha)*alpha);
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k = 0;
int first_j;
double * rB;
double * qB;
int j = 0;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================

*** After InlineDeviceFunctions  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx = blockIdx.x;
int tx = threadIdx.x;
if ((bx<number_boxes))
{
double a2 = ((2.0*alpha)*alpha);
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k = 0;
int first_j;
double * rB;
double * qB;
int j = 0;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[SeparateInitializers] begin
[SeparateInitializers] examining procedure kernel_gpu_cuda
[SeparateInitializers] end in 0.03 seconds
[LinkSymbol] 133 updates in 0.00 seconds

*** After SeparateInitializers  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx;
bx=blockIdx.x;
int tx;
tx=threadIdx.x;
if ((bx<number_boxes))
{
double a2;
a2=((2.0*alpha)*alpha);
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
k=0;
int first_j;
double * rB;
double * qB;
int j;
j=0;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[AnsiDeclarations] begin
[AnsiDeclarations] end in 0.00 seconds
[LinkSymbol] 133 updates in 0.00 seconds

*** After AnsiDeclarations  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx;
int tx;
bx=blockIdx.x;
tx=threadIdx.x;
if ((bx<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[StreamInsertion-FCUDA] begin
[StreamInsertion-FCUDA] examining procedure kernel_gpu_cuda
[StreamInsertion-FCUDA] end in 0.02 seconds
[LinkSymbol] 133 updates in 0.00 seconds

*** After StreamInsertion  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int bx;
int tx;
bx=blockIdx.x;
tx=threadIdx.x;
if ((bx<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[bx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[tx]=d_rv_gpu[((4*first_i)+tx)];
fA_shared[tx]=d_fv_gpu[((4*first_i)+tx)];
for (k=0; k<(1+d_box_gpu_nn[bx]); k ++ )
{
int wtx;
if ((k==0))
{
pointer=bx;
}
else
{
pointer=d_box_gpu_number[(((bx*26)+k)-1)];
}
first_j=d_box_gpu_offset[pointer];
rB_shared[tx]=d_rv_gpu[((4*first_j)+tx)];
qB_shared[tx]=d_qv_gpu[(first_j+tx)];
wtx=tx;
if ((wtx<100))
{
for (j=0; j<(4*100); j+=4)
{
r2=((((double)rA_shared[(4*wtx)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*wtx)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*wtx)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*wtx)+3)])*((double)rB_shared[(j+3)]))));
u2=(a2*r2);
vij=exp(( - u2));
fs=(2*vij);
d.x=(((double)rA_shared[((4*wtx)+1)])-((double)rB_shared[(j+1)]));
fxij=(fs*d.x);
d.y=(((double)rA_shared[((4*wtx)+2)])-((double)rB_shared[(j+2)]));
fyij=(fs*d.y);
d.z=(((double)rA_shared[((4*wtx)+3)])-((double)rB_shared[(j+3)]));
fzij=(fs*d.z);
fA_shared[(4*wtx)]+=((double)(((double)qB_shared[(j/4)])*vij));
fA_shared[((4*wtx)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
fA_shared[((4*wtx)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
fA_shared[((4*wtx)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
}
}
__syncthreads();
}
d_fv_gpu[((4*first_i)+tx)]=fA_shared[tx];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[RemoveThrDepLoops-FCUDA] begin
[RemoveThrDepLoops-FCUDA] examining procedure kernel_gpu_cuda
mVar2Var:
{d_box_gpu_nn=[], d_qv_gpu=[], fxij=[u2, threadIdx, alpha, fs, a2, vij, r2, j], fA=[], rB_shared=[first_j, threadIdx, pointer, k, blockIdx], d_rv_gpu=[], qB_shared=[first_j, threadIdx, pointer, k, blockIdx], first_j=[pointer, k, blockIdx], first_i=[blockIdx], rA_shared=[first_i, threadIdx, blockIdx], fA_shared=[first_i, fyij, u2, fzij, threadIdx, alpha, fs, a2, fxij, vij, r2, j, blockIdx], threadIdx=[], alpha=[], d_box_gpu_number=[], fs=[u2, threadIdx, alpha, a2, vij, r2, j], a2=[alpha], d_fv_gpu=[first_i, threadIdx, blockIdx], vij=[u2, threadIdx, alpha, a2, r2, j], d=[threadIdx, j], u2=[threadIdx, alpha, a2, r2, j], fzij=[u2, threadIdx, alpha, fs, a2, vij, r2, j], d_box_gpu_offset=[], j=[j], k=[k], rB=[], fyij=[u2, threadIdx, alpha, fs, a2, vij, r2, j], rA=[], qB=[], pointer=[k, blockIdx], r2=[threadIdx, j], number_boxes=[], blockIdx=[]}
[RemoveThrDepLoops-FCUDA] end in 0.03 seconds
[LinkSymbol] 154 updates in 0.00 seconds

*** After RemoveThrDepLoops  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[MakeArraysInCompute-FCUDA] begin
[MakeArraysInCompute-FCUDA] examining procedure kernel_gpu_cuda
Statement: {
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}
Statement: if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
Statement: {
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
Statement: double a2;
Statement: int first_i;
Statement: double * rA;
Statement: double * fA;
Statement: __shared__ double rA_shared[(4*100)];
Statement: __shared__ double fA_shared[(4*100)];
Statement: int pointer;
Statement: int k;
Statement: int first_j;
Statement: double * rB;
Statement: double * qB;
Statement: int j;
Statement: __shared__ double rB_shared[(4*100)];
Statement: __shared__ double qB_shared[100];
Statement: double r2;
Statement: double u2;
Statement: double vij;
Statement: double fs;
Statement: double fxij;
Statement: double fyij;
Statement: double fzij;
Statement: THREE_VECTOR d;
Statement: a2=((2.0*alpha)*alpha);
Statement: k=0;
Statement: j=0;
Statement: first_i=d_box_gpu_offset[blockIdx.x];
Statement: #pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
Statement: rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
Inside compute: rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
Statement: fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
Inside compute: fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
Statement: for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
Inside compute: for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
Statement: d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
Inside compute: d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
Statement: #pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
IDEXPR u2
[MakeArrays]: u2 is scalar:true used:false
IDEXPR fzij
[MakeArrays]: fzij is scalar:true used:false
IDEXPR fxij
[MakeArrays]: fxij is scalar:true used:false
WARNING: currently do not handle RHS elements of AccessExpression d in findVarsToConvert()
IDEXPR d
[MakeArrays]: d is scalar:true used:false
IDEXPR j
[MakeArrays]: j is scalar:true used:false
IDEXPR k
[MakeArrays]: k is scalar:true used:false
IDEXPR first_j
[MakeArrays]: first_j is scalar:true used:false
IDEXPR fyij
[MakeArrays]: fyij is scalar:true used:false
WARNING: currently do not handle RHS elements of AccessExpression d in findVarsToConvert()
IDEXPR d
[MakeArrays]: d is scalar:true used:false
WARNING: currently do not handle RHS elements of AccessExpression d in findVarsToConvert()
IDEXPR d
[MakeArrays]: d is scalar:true used:false
IDEXPR fs
[MakeArrays]: fs is scalar:true used:false
IDEXPR pointer
[MakeArrays]: pointer is scalar:true used:false
IDEXPR vij
[MakeArrays]: vij is scalar:true used:false
IDEXPR r2
[MakeArrays]: r2 is scalar:true used:false
Statement: k=0;
Statement: {
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
__syncthreads();
}
Statement: if ((k==0))
{
pointer=blockIdx.x;
}
else
{
pointer=d_box_gpu_number[(((blockIdx.x*26)+k)-1)];
}
Statement: first_j=d_box_gpu_offset[pointer];
Statement: rB_shared[threadIdx.x]=d_rv_gpu[((4*first_j)+threadIdx.x)];
Statement: qB_shared[threadIdx.x]=d_qv_gpu[(first_j+threadIdx.x)];
Statement: if ((threadIdx.x<100))
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
Statement: __syncthreads();
Statement: {
pointer=blockIdx.x;
}
Statement: {
pointer=d_box_gpu_number[(((blockIdx.x*26)+k)-1)];
}
Statement: {
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
Statement: pointer=blockIdx.x;
Statement: pointer=d_box_gpu_number[(((blockIdx.x*26)+k)-1)];
Statement: for (j=0; j<(4*100); j+=4)
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
Statement: j=0;
Statement: {
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
Statement: r2=((((double)rA_shared[(4*threadIdx.x)])+((double)rB_shared[j]))-(((((double)rA_shared[((4*threadIdx.x)+1)])*((double)rB_shared[(j+1)]))+(((double)rA_shared[((4*threadIdx.x)+2)])*((double)rB_shared[(j+2)])))+(((double)rA_shared[((4*threadIdx.x)+3)])*((double)rB_shared[(j+3)]))));
Statement: u2=(a2*r2);
Statement: vij=exp(( - u2));
Statement: fs=(2*vij);
Statement: d.x=(((double)rA_shared[((4*threadIdx.x)+1)])-((double)rB_shared[(j+1)]));
Statement: fxij=(fs*d.x);
Statement: d.y=(((double)rA_shared[((4*threadIdx.x)+2)])-((double)rB_shared[(j+2)]));
Statement: fyij=(fs*d.y);
Statement: d.z=(((double)rA_shared[((4*threadIdx.x)+3)])-((double)rB_shared[(j+3)]));
Statement: fzij=(fs*d.z);
Statement: fA_shared[(4*threadIdx.x)]+=((double)(((double)qB_shared[(j/4)])*vij));
Statement: fA_shared[((4*threadIdx.x)+1)]+=((double)(((double)qB_shared[(j/4)])*fxij));
Statement: fA_shared[((4*threadIdx.x)+2)]+=((double)(((double)qB_shared[(j/4)])*fyij));
Statement: fA_shared[((4*threadIdx.x)+3)]+=((double)(((double)qB_shared[(j/4)])*fzij));
[MakeArraysInCompute-FCUDA] end in 0.02 seconds
[LinkSymbol] 154 updates in 0.00 seconds

*** After MakeArraysInCompute  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda grid x_dim=128 
#pragma fcuda coreinfo num_cores=1 pipeline=yes 
#pragma fcuda portmerge port_id=0 remove_port_name=d_box_gpu_offset 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_nn 
#pragma fcuda portmerge port_id=1 remove_port_name=d_box_gpu_number 
#pragma fcuda portmerge port_id=2 remove_port_name=d_rv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_qv_gpu 
#pragma fcuda portmerge port_id=2 remove_port_name=d_fv_gpu 
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
__shared__ double rA_shared[(4*100)];
__shared__ double fA_shared[(4*100)];
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[SplitFcudaTasks-FCUDA] begin
[SplitFcudaTasks-FCUDA] examining procedure kernel_gpu_cuda
BRAM:rA_shared  specs: [[(4*100)]] size:1
BRAM:fA_shared  specs: [[(4*100)]] size:1
BRAM:rB_shared  specs: [[(4*100)]] size:1
BRAM:qB_shared  specs: [[100]] size:1

 ... Preprocessing pragma: 
	#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
Creating new FcudaCoreData for core: kernel_gpu_cuda_compute()

 ... Preprocessing pragma: 
	#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
fcudaCores (splitTasks-start):
[kernel_gpu_cuda_compute()]
coreNames: 
[kernel_gpu_cuda_compute()]
Checking Annotation Statement: [#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true ]
FCUDA compute begin=true level=2
Task stmt: rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
of type: class cetus.hir.ExpressionStatement
Task stmt: fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
of type: class cetus.hir.ExpressionStatement
Task stmt: for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
of type: class cetus.hir.ForLoop
Task stmt: d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
of type: class cetus.hir.ExpressionStatement
Task stmt: #pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
of type: class cetus.hir.AnnotationStatement
Checking Annotation Statement: [#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false ]
FCUDA compute begin=false level=2
Task use set: [a2, blockIdx.x, d.x, d.y, d.z, d_box_gpu_nn[blockIdx.x], d_box_gpu_number[(((blockIdx.x*26)+k)-1)], d_box_gpu_offset[pointer], d_fv_gpu[((4*first_i)+threadIdx.x)], d_qv_gpu[(first_j+threadIdx.x)], d_rv_gpu[((4*first_i)+threadIdx.x)], d_rv_gpu[((4*first_j)+threadIdx.x)], fA_shared[((4*threadIdx.x)+1)], fA_shared[((4*threadIdx.x)+2)], fA_shared[((4*threadIdx.x)+3)], fA_shared[(4*threadIdx.x)], fA_shared[threadIdx.x], first_i, first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared[(j/4)], r2, rA_shared[((4*threadIdx.x)+1)], rA_shared[((4*threadIdx.x)+2)], rA_shared[((4*threadIdx.x)+3)], rA_shared[(4*threadIdx.x)], rB_shared[(j+1)], rB_shared[(j+2)], rB_shared[(j+3)], rB_shared[j], threadIdx.x, u2, vij]
Task def set: [d.x, d.y, d.z, d_fv_gpu[((4*first_i)+threadIdx.x)], fA_shared[((4*threadIdx.x)+1)], fA_shared[((4*threadIdx.x)+2)], fA_shared[((4*threadIdx.x)+3)], fA_shared[(4*threadIdx.x)], fA_shared[threadIdx.x], first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared[threadIdx.x], r2, rA_shared[threadIdx.x], rB_shared[threadIdx.x], u2, vij]
Task maydef set: []
tmpExpr: a2
  of class: class cetus.hir.Identifier
decl: double a2
defStmt: none
tmpExpr: blockIdx.x
  of class: class cetus.hir.AccessExpression
WARNING: currently do not handle RHS elements of AccessExpression blockIdx.x in buildTaskVars()
tmpExpr: d.x
  of class: class cetus.hir.AccessExpression
WARNING: currently do not handle RHS elements of AccessExpression d.x in buildTaskVars()
decl: THREE_VECTOR d
defStmt: none
tmpExpr: d.y
  of class: class cetus.hir.AccessExpression
WARNING: currently do not handle RHS elements of AccessExpression d.y in buildTaskVars()
decl: THREE_VECTOR d
defStmt: none
tmpExpr: d.z
  of class: class cetus.hir.AccessExpression
WARNING: currently do not handle RHS elements of AccessExpression d.z in buildTaskVars()
decl: THREE_VECTOR d
defStmt: none
tmpExpr: d_box_gpu_nn[blockIdx.x]
  of class: class cetus.hir.ArrayAccess
decl: int * d_box_gpu_nn
defStmt: none
tmpExpr: d_box_gpu_number[(((blockIdx.x*26)+k)-1)]
  of class: class cetus.hir.ArrayAccess
decl: int * d_box_gpu_number
defStmt: none
tmpExpr: d_box_gpu_offset[pointer]
  of class: class cetus.hir.ArrayAccess
decl: long * d_box_gpu_offset
defStmt: none
tmpExpr: d_fv_gpu[((4*first_i)+threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: double * d_fv_gpu
defStmt: none
tmpExpr: d_qv_gpu[(first_j+threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: double * d_qv_gpu
defStmt: none
tmpExpr: d_rv_gpu[((4*first_i)+threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: double * d_rv_gpu
defStmt: none
tmpExpr: d_rv_gpu[((4*first_j)+threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: double * d_rv_gpu
defStmt: none
tmpExpr: fA_shared[((4*threadIdx.x)+1)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double fA_shared[(4*100)]
defStmt: none
tmpExpr: fA_shared[((4*threadIdx.x)+2)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double fA_shared[(4*100)]
defStmt: none
tmpExpr: fA_shared[((4*threadIdx.x)+3)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double fA_shared[(4*100)]
defStmt: none
tmpExpr: fA_shared[(4*threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double fA_shared[(4*100)]
defStmt: none
tmpExpr: fA_shared[threadIdx.x]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double fA_shared[(4*100)]
defStmt: none
tmpExpr: first_i
  of class: class cetus.hir.Identifier
decl: int first_i
defStmt: none
tmpExpr: first_j
  of class: class cetus.hir.Identifier
decl: int first_j
defStmt: none
tmpExpr: fs
  of class: class cetus.hir.Identifier
decl: double fs
defStmt: none
tmpExpr: fxij
  of class: class cetus.hir.Identifier
decl: double fxij
defStmt: none
tmpExpr: fyij
  of class: class cetus.hir.Identifier
decl: double fyij
defStmt: none
tmpExpr: fzij
  of class: class cetus.hir.Identifier
decl: double fzij
defStmt: none
tmpExpr: j
  of class: class cetus.hir.Identifier
decl: int j
defStmt: none
tmpExpr: k
  of class: class cetus.hir.Identifier
decl: int k
defStmt: none
tmpExpr: pointer
  of class: class cetus.hir.Identifier
decl: int pointer
defStmt: none
tmpExpr: qB_shared[(j/4)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double qB_shared[100]
defStmt: none
tmpExpr: qB_shared[threadIdx.x]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double qB_shared[100]
defStmt: none
tmpExpr: r2
  of class: class cetus.hir.Identifier
decl: double r2
defStmt: none
tmpExpr: rA_shared[((4*threadIdx.x)+1)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rA_shared[(4*100)]
defStmt: none
tmpExpr: rA_shared[((4*threadIdx.x)+2)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rA_shared[(4*100)]
defStmt: none
tmpExpr: rA_shared[((4*threadIdx.x)+3)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rA_shared[(4*100)]
defStmt: none
tmpExpr: rA_shared[(4*threadIdx.x)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rA_shared[(4*100)]
defStmt: none
tmpExpr: rA_shared[threadIdx.x]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rA_shared[(4*100)]
defStmt: none
tmpExpr: rB_shared[(j+1)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rB_shared[(4*100)]
defStmt: none
tmpExpr: rB_shared[(j+2)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rB_shared[(4*100)]
defStmt: none
tmpExpr: rB_shared[(j+3)]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rB_shared[(4*100)]
defStmt: none
tmpExpr: rB_shared[j]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rB_shared[(4*100)]
defStmt: none
tmpExpr: rB_shared[threadIdx.x]
  of class: class cetus.hir.ArrayAccess
decl: __shared__ double rB_shared[(4*100)]
defStmt: none
tmpExpr: threadIdx.x
  of class: class cetus.hir.AccessExpression
WARNING: currently do not handle RHS elements of AccessExpression threadIdx.x in buildTaskVars()
tmpExpr: u2
  of class: class cetus.hir.Identifier
decl: double u2
defStmt: none
tmpExpr: vij
  of class: class cetus.hir.Identifier
decl: double vij
defStmt: none
taskArgs: [enableSignal_compute, blockDim, gridDim, blockIdx, a2, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij]
taskDecls: [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, THREE_VECTOR d, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int first_j, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
defStmts: []
[SplitFcudaTasks-FCUDA] end in 0.03 seconds
[LinkSymbol] 181 updates in 0.00 seconds

*** After SplitFcudaTasks  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, THREE_VECTOR d, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int first_j, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij)
{
if (enableSignal_compute)
{
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double fA_shared[(4*100)];
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
double * rA;
double * fA;
int pointer;
int k;
int first_j;
double * rB;
double * qB;
int j;
double r2;
double u2;
double vij;
double fs;
double fxij;
double fyij;
double fzij;
THREE_VECTOR d;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[CleanKernelDecls-FCUDA] begin
[CleanKernelDecls-FCUDA] examining procedure kernel_gpu_cuda
cur_level:0
Defs+Uses:[blockIdx, blockIdx.x, blockIdx.y, enableSignal_compute, gridDim, gridDim.x, gridDim.y]
cur_level:1
Defs+Uses:[a2, alpha]
Defs+Uses:[k]
Defs+Uses:[j]
Defs+Uses:[blockIdx, blockIdx.x, d_box_gpu_offset, d_box_gpu_offset[blockIdx.x], first_i]
Defs+Uses:[]
Defs+Uses:[a2, blockDim, blockIdx, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, enableSignal_compute, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, gridDim, j, k, kernel_gpu_cuda_compute, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij]
Defs+Uses:[]
cur_level:1
var2freqMap{a2=2, d=1, fA=0, first_i=2, first_j=1, fs=1, fxij=1, fyij=1, fzij=1, j=2, k=2, pointer=1, qB=0, r2=1, rA=0, rB=0, u2=1, vij=1}
funcCallParams[a2, blockDim, blockIdx, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, enableSignal_compute, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, gridDim, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:d contains d
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, THREE_VECTOR d, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int first_j, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int first_j, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:first_j contains first_j
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int first_j, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, fs, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:fs contains fs
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fs, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, fxij, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:fxij contains fxij
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fxij, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, fyij, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:fyij contains fyij
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fyij, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, fzij, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:fzij contains fzij
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, double fzij, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:pointer contains pointer
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, int pointer, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, j, k, qB_shared, r2, rA_shared, rB_shared, u2, vij)
-arg:r2 contains r2
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], double r2, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, j, k, qB_shared, rA_shared, rB_shared, u2, vij)
-arg:u2 contains u2
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double u2, double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, j, k, qB_shared, rA_shared, rB_shared, vij)
-arg:vij contains vij
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)], double vij]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
cur_level:0
var2freqMap{enableSignal_compute=2, fA_shared=1, qB_shared=1, rA_shared=1, rB_shared=1}
funcCallParams[a2, blockDim, blockIdx, d, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, enableSignal_compute, fA_shared, first_i, first_j, fs, fxij, fyij, fzij, gridDim, j, k, pointer, qB_shared, r2, rA_shared, rB_shared, u2, vij]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, fA_shared, first_i, j, k, qB_shared, rA_shared, rB_shared)
-arg:fA_shared contains fA_shared
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, __shared__ double fA_shared[(4*100)], int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k, qB_shared, rA_shared, rB_shared)
-arg:qB_shared contains qB_shared
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double qB_shared[100], __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k, rA_shared, rB_shared)
-arg:rA_shared contains rA_shared
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double rA_shared[(4*100)], __shared__ double rB_shared[(4*100)]]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double rB_shared[(4*100)]]
fcall:kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k, rB_shared)
-arg:rB_shared contains rB_shared
- and are equal
- declList b4 = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k, __shared__ double rB_shared[(4*100)]]
- declList after = [int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k]
[CleanKernelDecls-FCUDA] end in 0.02 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After CleanKernelDecls  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
if (enableSignal_compute)
{
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[SerializeThreads-MCUDA] begin
[SerializeThreads-MCUDA] examining procedure kernel_gpu_cuda
[SerializeThreads-MCUDA] end in 0.01 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After SerializeThreads  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
if (enableSignal_compute)
{
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
for (k=0; k<(1+d_box_gpu_nn[blockIdx.x]); k ++ )
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
__syncthreads();
}
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[EnforceSyncs-MCUDA] begin
[EnforceSyncs-MCUDA] examining procedure kernel_gpu_cuda
[EnforceSyncs-MCUDA] end in 0.02 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After EnforceSyncs  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[PrivatizeScalarsInThreadLoops - FCUDA] begin
[PrivatizeScalarsInThreadLoops - FCUDA] examining procedure kernel_gpu_cuda
THREADLOOP: {
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
THREADLOOP: {
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
k=0;
}
THREADLOOP: {
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
THREADLOOP: {
k ++ ;
}
DEFSET: [r2, u2, vij, fs, <d.x>, fxij, <d.y>, fyij, <d.z>, fzij]
Find UseSet of: {
rA_shared[threadIdx.x]=d_rv_gpu[((4*first_i)+threadIdx.x)];
fA_shared[threadIdx.x]=d_fv_gpu[((4*first_i)+threadIdx.x)];
k=0;
}
USESET: [* d_rv_gpu, * d_fv_gpu, first_i, <threadIdx.x>]
DEFSET WITHIN LOOP: [rA_shared[(4*100)], fA_shared[(4*100)], k]
REMAINSET: []
Find UseSet of: {
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
USESET: [first_i, fA_shared[(4*100)], <threadIdx.x>]
DEFSET WITHIN LOOP: [* d_fv_gpu]
REMAINSET: []
Find UseSet of: {
k ++ ;
}
USESET: [k]
DEFSET WITHIN LOOP: [k]
REMAINSET: []
Scalars to be privatized: []
[PrivatizeScalarsInThreadLoops - FCUDA] end in 0.03 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After PrivatizeScalarsInThreadLoop  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[UnrollThreadLoops-MCUDA] begin
[UnrollThreadLoops-MCUDA] examining procedure kernel_gpu_cuda

[Unrolling] : kernel_gpu_cuda_compute
[Proc]: #pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
}
}


[unrollFactor] 1
mUnrolledIDs: 
{}
[UnrollThreadLoops-MCUDA] end in 0.00 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After UnrollThreadLoops  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[PartitionArrays-MCUDA] begin
[PartitionArrays-MCUDA] examining procedure kernel_gpu_cuda
[numDims]1
[Memory partition] : kernel_gpu_cuda_compute

[Proc]: #pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
}
}


[mempartFactor]1
[Memory partition] : kernel_gpu_cuda

HAA 1 {
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}
[PartitionArrays-MCUDA] end in 0.00 seconds
[LinkSymbol] 167 updates in 0.00 seconds

*** After PartitionArrays  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
double a2;
int first_i;
int k;
int j;
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=true unroll=1 begin=false 
}
}



===========================================
[IfSplitPass-FCUDA] begin
[IfSplitPass-FCUDA] examining procedure kernel_gpu_cuda
fcudaCores:
[kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k)]
coreNames: 
[kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k)]
Handling control flow for kernel_gpu_cuda_compute(enableSignal_compute, blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k)
mCurrEnableSignal: enableSignal_compute
In if stmt, lead list {
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
}

In if stmt, trail list {

}
[IfSplitPass-FCUDA] end in 0.00 seconds
[LinkSymbol] 170 updates in 0.00 seconds

*** After IfSplitPass  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu)
{
int enableSignal_compute;
dim3 blockIdx;
double a2;
int first_i;
int k;
int j;
enableSignal_compute=((blockIdx.x<gridDim.x)&&(blockIdx.y<gridDim.y));
if ((blockIdx.x<number_boxes))
{
a2=((2.0*alpha)*alpha);
k=0;
j=0;
first_i=d_box_gpu_offset[blockIdx.x];
}
kernel_gpu_cuda_compute((enableSignal_compute&&(blockIdx.x<number_boxes)), blockDim, gridDim, blockIdx, a2, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_fv_gpu, d_qv_gpu, d_rv_gpu, first_i, j, k);
}



===========================================
[WrapBlockIdxLoop-FCUDA] begin
[WrapBlockIdxLoop-FCUDA] examining procedure kernel_gpu_cuda
[WrapBlockIdxLoop-FCUDA] end in 0.00 seconds
[LinkSymbol] 170 updates in 0.00 seconds

*** After WrapBlockIdxLoop  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
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



===========================================
[CleanThreadLoops-MCUDA] begin
[CleanThreadLoops-MCUDA] examining procedure kernel_gpu_cuda
[CleanThreadLoops-MCUDA] end in 0.00 seconds
[LinkSymbol] 170 updates in 0.00 seconds

*** After CleanThreadLoops  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
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



===========================================
[KernelStateTransform-MCUDA] begin
[KernelStateTransform-MCUDA] examining procedure kernel_gpu_cuda
>>> outside uses:
[a2, alpha, blockIdx.x, blockIdx.y, d_box_gpu_nn, d_box_gpu_number, d_box_gpu_offset, d_box_gpu_offset[blockIdx.x], d_fv_gpu, d_qv_gpu, d_rv_gpu, enableSignal_compute, first_i, gridDim.x, gridDim.y, j, k, number_boxes]
>>> handling: a2
>>> handling: alpha
>>> handling: blockIdx
>>> handling: blockIdx
>>> handling: * d_box_gpu_nn
>>> handling: * d_box_gpu_number
>>> handling: * d_box_gpu_offset
>>> handling: * d_box_gpu_offset
>>> handling: * d_fv_gpu
>>> handling: * d_qv_gpu
>>> handling: * d_rv_gpu
>>> handling: enableSignal_compute
>>> handling: first_i
>>> handling: gridDim
>>> handling: gridDim
>>> handling: j
>>> handling: k
>>> handling: number_boxes
transforming Decls
[KernelStateTransform-MCUDA] end in 0.01 seconds
[LinkSymbol] 170 updates in 0.00 seconds

*** After KernelStateTransform  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__syncthreads();
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
k ++ ;
}
__syncthreads();
}
for (threadIdx.x=0;threadIdx.x<blockDim.x ; threadIdx.x=threadIdx.x+1) 
{
d_fv_gpu[((4*first_i)+threadIdx.x)]=fA_shared[threadIdx.x];
}
__syncthreads();
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
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



===========================================
[CleanSyncFunc-MCUDA] begin
[CleanSyncFunc-MCUDA] examining procedure kernel_gpu_cuda
[CleanSyncFunc-MCUDA] end in 0.00 seconds
[LinkSymbol] 169 updates in 0.00 seconds

*** After CleanSyncFunc  ***
#include <fcuda.h>
#include "main.h"
#include <string.h>
#include <math.h>
const int BLOCKDIM_X_kernel_gpu_cuda = 128;
#pragma fcuda compute array_split=[] mpart=1 name=compute cores=[1] end=false unroll=1 begin=true 
void kernel_gpu_cuda_compute(int enableSignal_compute, dim3 blockDim, dim3 gridDim, dim3 blockIdx, double a2, int * d_box_gpu_nn, int * d_box_gpu_number, long * d_box_gpu_offset, double * d_fv_gpu, double * d_qv_gpu, double * d_rv_gpu, int first_i, int j, int k)
{
dim3 __shared__ threadIdx;
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
__shared__ double fA_shared[(4*100)];
__shared__ double qB_shared[100];
__shared__ double rA_shared[(4*100)];
__shared__ double rB_shared[(4*100)];
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
__global__ void kernel_gpu_cuda(double alpha, long number_boxes, long * d_box_gpu_offset, int * d_box_gpu_nn, int * d_box_gpu_number, double * d_rv_gpu, double * d_qv_gpu, double * d_fv_gpu, dim3 gridDim, dim3 blockDim, int num_cores, int core_id)
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



===========================================
