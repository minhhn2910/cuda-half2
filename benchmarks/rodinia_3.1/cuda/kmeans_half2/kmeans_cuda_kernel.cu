#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<int, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<int, 1, cudaReadModeElementType> t_features_flipped;
texture<int, 1, cudaReadModeElementType> t_clusters;


__constant__ half c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ...
   [p1,dim0][p1,dim1][p1,dim2] ...
   [p2,dim0][p2,dim1][p2,dim2] ...
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(half *input,			/* original */
							   half *output,			/* inverted */
							   int npoints,				/* npoints */
							   int nfeatures)			/* nfeatures */
{
	int point_id = threadIdx.x + blockDim.x*blockIdx.x;	/* id of thread */
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
	}
	return;
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION


/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void
kmeansPoint(half2  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			half2  *clusters,
			half2  *block_clusters,
			int    *block_deltas)
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	// point/thread ID
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;

	int  index = -1;
	int index2 = -1;
	if (point_id < npoints/2)
	{
		int i, j;
		half2 min_dist = __float2half2_rn(FLT_MAX);
		half2 dist;													/* distance square between a point to cluster center */

		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */
			half2 ans=__float2half2_rn(0.0);												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				int temp = tex1Dfetch(t_features,addr);
				half2 diff = ( *(__half2*)(&temp)- __half2half2(c_clusters[cluster_base_index + j]));	/* distance between a data point to cluster centers */
				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (((half*)(&dist))[0] < ((half*)(&min_dist))[0]) {
				((half*)(&min_dist))[0] = ((half*)(&dist))[0] ;
				index    = i;
			}
			if (((half*)(&dist))[1] < ((half*)(&min_dist))[1]) {
				((half*)(&min_dist))[1] = ((half*)(&dist))[1] ;
				index2    = i;
			}
	}


		/* assign the membership to object point_id */
		membership[2*point_id] = index;
		membership[2*point_id+1] = index2;
	}

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
