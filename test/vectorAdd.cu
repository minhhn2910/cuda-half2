//~ #include <half.hpp>
__device__ void Vec_add(float *x, float *y , float* z, float gaaa[], int n) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int i = blockDim.x * blockIdx.x + threadIdx.x;
	float test_val, test_val2;
   /* block_count*threads_per_block may be >= n */
   if (i < n) {


	z[i] = x[i] + y[i+2]  + gaaa[(2*3+i)+456];

		return;
   }
}  /* Vec_add */
