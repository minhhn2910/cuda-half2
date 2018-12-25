#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<short, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<short, 1, cudaReadModeElementType> t_features_flipped;
texture<short, 1, cudaReadModeElementType> t_clusters;


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
kmeansPoint(half  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			half  *clusters,
			half  *block_clusters,
			int    *block_deltas)
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	// point/thread ID
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;

	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		half min_dist = __float2half(65535);
		half dist;													/* distance square between a point to cluster center */

		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */
			half ans=__float2half(0.0);												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				short temp = tex1Dfetch(t_features,addr);
				half diff = (*((half*)(&temp)) -
							  c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */

				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}

		membership[point_id] = index;
	}

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
