/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster

	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by

	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science

***********************************************/
#include "streamcluster_header.cu"
#include "newhalf.hpp"
#include <cuda_fp16.h>

#include "half2_operator_overload.cuh"
#include "half_operator_overload.cuh"
typedef struct {
  half2 weight;
  float *coord;
  long2 assign;  /* number of point where this one is assigned */
  half2 cost;  /* cost of that assignment, weight*distance */
} Point_dev_half2;

int counting = 0;
using namespace std;

// AUTO-ERROR CHECK FOR ALL CUDA FUNCTIONS
#define CUDA_SAFE_CALL( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define CUDATIME

// host memory
float *work_mem_h;
float *coord_h;

half_float::half *work_mem_half;
half_float::half *coord_half;

// device memory
half2 *work_mem_d;
half2 *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point_dev_half2 *p;

static int iter = 0;		// counter for total# of iteration


//=======================================
// Euclidean Distance
//=======================================
__device__ __half2
d_dist(int p1, int p2, int num, int dim, __half2 *coord_d)
{
	__half2 retval = __float2half2_rn(0.0);
	for(int i = 0; i < dim; i++){
		//~ __half2 tmp = coord_d[(i*num/2)+p1] - coord_d[(i*num/2)+p2/2];
		__half2 tmp = coord_d[(i*num/2)+p1] - __half2half2(((half*)coord_d)[(i*num)+p2]);
		retval += tmp * tmp;
	}
	return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void
kernel_compute_cost(int num, int dim, long x, Point_dev_half2 *p, int K, int stride,
					__half2 *coord_d, __half2 *work_mem_d, int *center_table_d, bool *switch_membership_d)
{
	// block ID and global thread ID
	const int bid  = blockIdx.x + gridDim.x * blockIdx.y;
	const int tid = blockDim.x * bid + threadIdx.x;

	if(tid < num/2)
	{
		__half *lower1 = &((__half*)work_mem_d)[tid*2*stride]; //half array
		__half *lower2 = &((__half*)work_mem_d)[(tid*2+1)*stride]; //half array
		

		// cost between this point and point[x]: euclidean distance multiplied by weight
		__half2 x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;

		//~ work_mem_d[tid] = x_cost;
		
		// if computed cost is less then original (it saves), mark it as to reassign
    float2 x_cost_temp = __half22float2(x_cost);
    float2 p_cost_temp = __half22float2(p[tid].cost);
//test
  /*   lower[K] += __float2half(1);
		lower[center_table_d[p[tid].assign.y]] = __float2half(1);
		lower[center_table_d[p[tid].assign.x]] = __float2half(1);
		lower[center_table_d[p[tid].assign.y]] = __float2half(1);
		lower[center_table_d[p[tid].assign.x]] = __float2half(1);
switch_membership_d[2*tid] = 1;
switch_membership_d[2*tid+1] = 1;
*/
		if ( x_cost_temp.x < p_cost_temp.x)
		{
			switch_membership_d[2*tid] = 1;
			lower1[K] += __float2half(x_cost_temp.x - p_cost_temp.x);
		}
		// if computed cost is larger, save the difference
		else
		{
			lower1[center_table_d[p[tid].assign.x]] += __float2half(p_cost_temp.x - x_cost_temp.x);
		}
		if ( x_cost_temp.y < p_cost_temp.y)
		{
			switch_membership_d[2*tid+1] = 1;
			lower2[K] += __float2half(x_cost_temp.y - p_cost_temp.y);
		}
		// if computed cost is larger, save the difference
		else
		{
			lower2[center_table_d[p[tid].assign.y]] += __float2half(p_cost_temp.y - x_cost_temp.y);
		}

  /*
		if ( x_cost < p[tid].cost )
		{
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - p[tid].cost;
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
		}*/

	}
}

//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim)
{
	CUDA_SAFE_CALL( cudaMalloc((void**) &center_table_d,	  num * sizeof(int))   );
	CUDA_SAFE_CALL( cudaMalloc((void**) &switch_membership_d, num * sizeof(bool))  );
	CUDA_SAFE_CALL( cudaMalloc((void**) &p,					  num/2 * sizeof(Point_half2)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &coord_d,		num * dim * sizeof(__half)) );
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim)
{
	coord_h	= (float*) malloc( num * dim * sizeof(float) );
  coord_half	= (half_float::half*) malloc( num * dim * sizeof(half) );
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem()
{
	CUDA_SAFE_CALL( cudaFree(center_table_d)	  );
	CUDA_SAFE_CALL( cudaFree(switch_membership_d) );
	CUDA_SAFE_CALL( cudaFree(p)					  );
	CUDA_SAFE_CALL( cudaFree(coord_d)			  );
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem()
{
	free(coord_h);
  free(coord_half);

}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged,
							double *serial_t, double *cpu_to_gpu_t, double *gpu_to_cpu_t, double *alloc_t, double *kernel_t, double *free_t)
{
#ifdef CUDATIME
	float tmp_t;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif

	cudaError_t error;

	int stride	= *numcenters + 1;			// size of each work_mem segment
	int K		= *numcenters ;				// number of centers
	int num		=  points->num;				// number of points
	int dim		=  points->dim;				// number of dimension
	int nThread =  num;						// number of threads == number of data points

	//=========================================
	// ALLOCATE HOST MEMORY + DATA PREPARATION
	//=========================================
	work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float) );

  //printf ("num : %d\n",num);

  work_mem_half = (half_float::half*) malloc(stride * (nThread + 1) * sizeof(half) );

	// Only on the first iteration
	if(iter == 0)
	{
		allocHostMem(num, dim);
	}

	// build center-index table
	int count = 0;
	for( int i=0; i<num; i++)
	{
		if( is_center[i] )
		{
			center_table[i] = count++;
		}
	}

	// Extract 'coord'
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		for(int i=0; i<dim; i++)
		{
			for(int j=0; j<num; j++)
			{
				coord_h[ (num*i)+j ] = points->p[j].coord[i];
        coord_half[ (num*i)+j ] = half_float::half(coord_h[ (num*i)+j ]);
			}
		}
	}
/*typedef struct {
  half_float::half weight;
  half_float::half *coord;
  long assign;
  half_float::half cost;
} Point_half;*/
  //copy points->p
  //points.num = chunksize;
  //points.p = (Point *)malloc(chunksize*sizeof(Point))
  Point_half2* point_half =(Point_half2*) malloc(points->num/2*sizeof(Point_half2));
  for(int i = 0;i <points->num/2; i++){
      point_half[i].weight = floats2half2(points->p[2*i+1].weight,points->p[2*i].weight);

      point_half[i].assign = make_long2(points->p[2*i].assign, points->p[2*i+1].assign); /// in reverse order

      point_half[i].cost = floats2half2(points->p[2*i+1].cost,points->p[2*i].cost);

  }
  //~ //printf("%ld %ld ",point_half[0].assign.x, point_half[0].assign.y);

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif

	//=======================================
	// ALLOCATE GPU MEMORY
	//=======================================
	CUDA_SAFE_CALL( cudaMalloc((void**) &work_mem_d,  stride * (nThread + 1) * sizeof(half)) );
	// Only on the first iteration
	if( iter == 0 )
	{
		allocDevMem(num, dim);
	}

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*alloc_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif

	//=======================================
	// CPU-TO-GPU MEMORY COPY
	//=======================================
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		CUDA_SAFE_CALL( cudaMemcpy(coord_d,  coord_half,	 num * dim * sizeof(half), cudaMemcpyHostToDevice) );
	}
	CUDA_SAFE_CALL( cudaMemcpy(center_table_d,  center_table,  num * sizeof(int),   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(p,  point_half,				   num/2 * sizeof(Point_half2), cudaMemcpyHostToDevice) );

	CUDA_SAFE_CALL( cudaMemset((void*) switch_membership_d, 0,			num * sizeof(bool))  );
	CUDA_SAFE_CALL( cudaMemset((void*) work_mem_d,  		0, stride * (nThread + 1) * sizeof(half)) );

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*cpu_to_gpu_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif
	printf("call me %d  num %d\n", counting++, num);
	//=======================================
	// KERNEL: CALCULATE COST
	//=======================================
	// Determine the number of thread blocks in the x- and y-dimension
	int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
	int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
	int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	//~ printf("%d %d %d stride %d k %d \n",num_blocks,num_blocks_y,num_blocks_x, stride, K);  

	kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK/2>>>(
															num,					// in:	# of data
															dim,					// in:	dimension of point coordinates
															x,						// in:	point to open a center at
															p,						// in:	data point array
															K,						// in:	number of centers
															stride,					// in:  size of each work_mem segment
															coord_d,				// in:	array of point coordinates
															work_mem_d,				// out:	cost and lower field array
															center_table_d,			// in:	center index table
															switch_membership_d		// out:  changes in membership
															);
	cudaThreadSynchronize();

	// error check
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*kernel_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif

	//=======================================
	// GPU-TO-CPU MEMORY COPY
	//=======================================
	CUDA_SAFE_CALL( cudaMemcpy(work_mem_half, 		  work_mem_d, 	stride * (nThread + 1) * sizeof(half), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaMemcpy(switch_membership, switch_membership_d,	 num * sizeof(bool),  cudaMemcpyDeviceToHost) );

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*gpu_to_cpu_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif

  for(int i = 0 ; i<stride * (nThread + 1) ; i++){
    work_mem_h[i] = float(work_mem_half[i]);
    //~ printf("%f, ",work_mem_h[i]);
  }
   //~ printf ("\n");
   
     //~ for (int i = 0 ; i < num; i++){
	  //~ printf("%d, ", switch_membership[i]);
	  //~ }
	//~ printf ("\n"); 
	//=======================================
	// CPU (SERIAL) WORK
	//=======================================
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem_h[stride * nThread];
	// compute the number of centers to close if we are to open i
	for(int i=0; i < num; i++)
	{
		if( is_center[i] )
		{
			float low = z;
		    for( int j = 0; j < num; j++ )
			{
				low += work_mem_h[ j*stride + center_table[i] ];
			}

		    gl_lower[center_table[i]] = low;

		    if ( low > 0 )
			{
				++number_of_centers_to_close;
				work_mem_h[i*stride+K] -= low;
		    }
		}
		gl_cost_of_opening_x += work_mem_h[i*stride+K];
	}

	//if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
	if ( gl_cost_of_opening_x < 0 )
	{
		for(int i = 0; i < num; i++)
		{
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
			if ( switch_membership[i] || close_center )
			{
				points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
				points->p[i].assign = x;
			}
		}

		for(int i = 0; i < num; i++)
		{
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
			{
				is_center[i] = false;
			}
		}

		if( x >= 0 && x < num)
		{
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	}
	else
	{
		gl_cost_of_opening_x = 0;
	}

	//=======================================
	// DEALLOCATE HOST MEMORY
	//=======================================
	free(work_mem_h);
  free(work_mem_half);
  free(point_half);

#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;

	cudaEventRecord(start,0);
#endif

	//=======================================
	// DEALLOCATE GPU MEMORY
	//=======================================
	CUDA_SAFE_CALL( cudaFree(work_mem_d) );


#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*free_t += (double) tmp_t;
#endif
	iter++;
	return -gl_cost_of_opening_x;
}
