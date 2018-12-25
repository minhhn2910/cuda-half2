//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/device/device.h"				// (in library path specified to compiler)	needed by for device functions
#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

#include <cuda_fp16.h>
#include "../newhalf.hpp"
#include "../half_operator_overload.cuh"
#include "../half2_operator_overload.cuh"

typedef struct
{
	half x, y, z;

} THREE_VECTOR_HALF;

typedef struct
{
	half v, x, y, z;

} FOUR_VECTOR_HALF;


typedef struct
{
	half_float::half x, y, z;

} THREE_VECTOR_HALF_HOST;

typedef struct
{
	half_float::half v, x, y, z;

} FOUR_VECTOR_HALF_HOST;

typedef struct
{
	half2 x, y, z;//x.high, x.low, ....

} THREE_VECTOR_HALF2;
typedef struct
{
	half2 v, x, y, z;

} FOUR_VECTOR_HALF2;

typedef struct
{
	uint32_t x, y, z;

} THREE_VECTOR_HALF2_HOST;

typedef struct
{
	uint32_t v, x, y, z;

} FOUR_VECTOR_HALF2_HOST;

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	cudaThreadSynchronize();

	//====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	
	
	FOUR_VECTOR_HALF2* d_rv_gpu;
	half2* d_qv_gpu;
	FOUR_VECTOR_HALF2* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	cudaMalloc(	(void **)&d_box_gpu, 
				dim_cpu.box_mem);

	//==================================================50
	//	rv
	//==================================================50

	cudaMalloc(	(void **)&d_rv_gpu, 
				dim_cpu.space_mem/2);

	//==================================================50
	//	qv
	//==================================================50

	cudaMalloc(	(void **)&d_qv_gpu, 
				dim_cpu.space_mem2/2);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	cudaMalloc(	(void **)&d_fv_gpu, 
				dim_cpu.space_mem/2);

	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

//convert before copying dim_cpu.space_elem 
						FOUR_VECTOR_HALF2_HOST* rv_cpu_half = (FOUR_VECTOR_HALF2_HOST*) malloc(dim_cpu.space_mem/2);
						half_float::half* qv_cpu_half = (half_float::half*) malloc(dim_cpu.space_mem2/2);
						FOUR_VECTOR_HALF2_HOST* fv_cpu_half = (FOUR_VECTOR_HALF2_HOST*) malloc(dim_cpu.space_mem/2);
						
	//~ int i;
	//~ for(i=0; i<dim_cpu.space_elem; i=i+1){
		//~ rv_cpu_half[i].v = half_float::half(rv_cpu[i].v);
		//~ rv_cpu_half[i].x = half_float::half(rv_cpu[i].x);			// get a number in the range 0.1 - 1.0
		//~ rv_cpu_half[i].y = half_float::half(rv_cpu[i].y);				// get a number in the range 0.1 - 1.0
		//~ rv_cpu_half[i].z = half_float::half(rv_cpu[i].z);			// get a number in the range 0.1 - 1.0
		
		//~ fv_cpu_half[i].v = half_float::half(fv_cpu[i].v);
		//~ fv_cpu_half[i].x = half_float::half(fv_cpu[i].x);			// get a number in the range 0.1 - 1.0
		//~ fv_cpu_half[i].y = half_float::half(fv_cpu[i].y);				// get a number in the range 0.1 - 1.0
		//~ fv_cpu_half[i].z = half_float::half(fv_cpu[i].z);			// get a number in the range 0.1 - 1.0
	//~ }

	int i;
	for(i=0; i<dim_cpu.space_elem/2; i=i+1){ //need to convert
		//rv_cpu_half[i].v = rv_cpu[i].v;
		//uint32_t haf2_valv = std::floats2half2(low_val,high_val);
		rv_cpu_half[i].v  = floats2half2(rv_cpu[i*2+1].v,rv_cpu[i*2].v);
		rv_cpu_half[i].x = floats2half2(rv_cpu[i*2+1].x,rv_cpu[i*2].x);
		rv_cpu_half[i].y = floats2half2(rv_cpu[i*2+1].y,rv_cpu[i*2].y);
		rv_cpu_half[i].z = floats2half2(rv_cpu[i*2+1].z,rv_cpu[i*2].z);
		
		fv_cpu_half[i].v  = floats2half2(fv_cpu[i*2+1].v,fv_cpu[i*2].v);
		fv_cpu_half[i].x = floats2half2(fv_cpu[i*2+1].x,fv_cpu[i*2].x);
		fv_cpu_half[i].y = floats2half2(fv_cpu[i*2+1].y,fv_cpu[i*2].y);
		fv_cpu_half[i].z = floats2half2(fv_cpu[i*2+1].z,fv_cpu[i*2].z);
	}
	
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu_half[i] = half_float::half(qv_cpu[i]);			
	}

	cudaMemcpy(	d_box_gpu, 
				box_cpu,
				dim_cpu.box_mem, 
				cudaMemcpyHostToDevice);

	//==================================================50
	//	rv
	//==================================================50

	cudaMemcpy(	d_rv_gpu,
				rv_cpu_half,
				dim_cpu.space_mem/2,
				cudaMemcpyHostToDevice);

	//==================================================50
	//	qv
	//==================================================50

	cudaMemcpy(	d_qv_gpu,
				qv_cpu_half,
				dim_cpu.space_mem2/2,
				cudaMemcpyHostToDevice);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	cudaMemcpy(	d_fv_gpu, 
				fv_cpu_half, 
				dim_cpu.space_mem/2, 
				cudaMemcpyHostToDevice);

	time3 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads>>>(	par_cpu,
											dim_cpu,
											d_box_gpu,
											d_rv_gpu,
											d_qv_gpu,
											d_fv_gpu);

	checkCUDAError("Start");
	cudaThreadSynchronize();

	time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	cudaMemcpy(	fv_cpu_half, 
				d_fv_gpu, 
				dim_cpu.space_mem/2, 
				cudaMemcpyDeviceToHost);

//convert back
	for(i=0; i<dim_cpu.space_elem/2; i=i+1){ //need to convert
		//rv_cpu_half[i].v = rv_cpu[i].v;
		//uint32_t haf2_valv = std::floats2half2(low_val,high_val);
		fv_cpu[i*2].v = half2high2float(fv_cpu_half[i].v);
		fv_cpu[i*2+1].v =  half2low2float(fv_cpu_half[i].v);
		
		fv_cpu[i*2].x = half2high2float(fv_cpu_half[i].x);
		fv_cpu[i*2+1].x =  half2low2float(fv_cpu_half[i].x);
		
		fv_cpu[i*2].y = half2high2float(fv_cpu_half[i].y);
		fv_cpu[i*2+1].y =  half2low2float(fv_cpu_half[i].y);
		
		fv_cpu[i*2].z = half2high2float(fv_cpu_half[i].z);
		fv_cpu[i*2+1].z =  half2low2float(fv_cpu_half[i].z);

	}
	
	//~ for(i=0; i<dim_cpu.space_elem/4; i=i+1){ 
	//~ printf( "%d %f %f %f %f\n", i, fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
	//~ }

	time5 = get_time();







	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	cudaFree(d_rv_gpu);
	cudaFree(d_qv_gpu);
	cudaFree(d_fv_gpu);
	cudaFree(d_box_gpu);

	time6 = get_time();
printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);


	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150
	//~ printf ("test teST test");
	/*
	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time6-time0) / 1000000);
	*/
}
