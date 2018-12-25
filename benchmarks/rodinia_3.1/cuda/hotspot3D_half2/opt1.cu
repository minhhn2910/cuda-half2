long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#include "newhalf.hpp"
#include <cuda_fp16.h>
#include "half2_operator_overload.cuh"
__global__ void hotspotOpt1(__half2 *p, __half2* tIn, __half2 *tOut, __half2 sdc,
        int nx, int ny, int nz,
        __half2 ce, __half2 cw,
        __half2 cn, __half2 cs,
        __half2 ct, __half2 cb,
        __half2 cc)
{
    __half2 amb_temp = __float2half2_rn(80.0);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = (i*2 + j * nx);
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx/2-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    __half2 temp1, temp2, temp3;
    temp1 = temp2 = tIn[c/2];
    temp3 = tIn[(c+xy)/2];
    tOut[c/2] = cc * temp2 + cw * tIn[W/2] + ce * tIn[E/2] + cs * tIn[S/2]
        + cn * tIn[N/2] + cb * temp1 + ct * temp3 + sdc * p[c/2] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[(c+xy)/2];
        tOut[c/2] = cc * temp2 + cw * tIn[W/2] + ce * tIn[E/2] + cs * tIn[S/2]
            + cn * tIn[N/2] + cb * temp1 + ct * temp3 + sdc * p[c/2] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c/2] = cc * temp2 + cw * tIn[W/2] + ce * tIn[E/2] + cs * tIn[S/2]
        + cn * tIn[N/2] + cb * temp1 + ct * temp3 + sdc * p[c/2] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap,
        float Rx, float Ry, float Rz,
        float dt, int numiter)
{
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;

		half_float::half* p_half;
		half_float::half* tIn_half;
		half_float::half* tOut_half;

		size_t half_size = sizeof(half)* nx * ny * nz;
		p_half = (half_float::half*)malloc(half_size);
		tIn_half =(half_float::half*)malloc(half_size);
		tOut_half = (half_float::half*)malloc(half_size);

    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(half) * nx * ny * nz;

		for(int i = 0; i < nx * ny * nz; i++ ){

			p_half[i] = p[i];
			tIn_half[i] = tIn[i];
		}



    __half2  *tIn_d, *tOut_d, *p_d;

		printf ("val: %f %f %f %f %f %f %f %f \n",stepDivCap,ce,cw,cn,ct,cb,cc);
    cudaMalloc((void**)&p_d,s);
    cudaMalloc((void**)&tIn_d,s);
    cudaMalloc((void**)&tOut_d,s);
    cudaMemcpy(tIn_d, tIn_half, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p_half, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx /2 /64, ny / 4, 1);

		uint32_t stepDivCap_half = floats2half2 (stepDivCap,stepDivCap );
		__half2 stdc = *(__half2*)&(stepDivCap_half);

		uint32_t ce_half = floats2half2 (ce ,ce);
		__half2 ce_dev = *(__half2*)&ce_half;

		uint32_t cw_half = floats2half2 (cw, cw);
		__half2 cw_dev = *(__half2*)&cw_half;

		uint32_t cn_half = floats2half2 (cn, cn);
		__half2 cn_dev = *(__half2*)&cn_half;

		uint32_t cs_half = floats2half2 (cs, cs);
		__half2 cs_dev = *(__half2*)&cs_half;

		uint32_t ct_half = floats2half2 (ct, ct);
		__half2 ct_dev = *(__half2*)&ct_half;

		uint32_t cb_half = floats2half2 (cb, cb);
		__half2 cb_dev = *(__half2*)&cb_half;

		uint32_t cc_half = floats2half2 (cc,cc);
		__half2 cc_dev = *(__half2*)&cc_half;
float time_kernel = 0.0;
float tmp_t;
    long long start = get_time();
    cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	cudaEventRecord(start_event,0);
	
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (p_d, tIn_d, tOut_d, stdc, nx, ny, nz, ce_dev, cw_dev, cn_dev, cs_dev, ct_dev, cb_dev, cc_dev);
        __half2 *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
/*
		for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (p_d, tIn_d, tOut_d, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
        __half2 *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
*/
    cudaDeviceSynchronize();
    
     cudaThreadSynchronize();
	cudaEventRecord(stop_event,0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&tmp_t, start_event, stop_event);
	time_kernel += tmp_t;

    long long stop = get_time();
    float time = (float)((stop - start)/(1000.0 * 1000.0));
    printf("Time: %.3f (s)\n",time);
    printf("Time kernel: %.3f (s)\n",time_kernel);
    cudaMemcpy(tOut_half, tOut_d, s, cudaMemcpyDeviceToHost);

		for(int i = 0; i < nx * ny * nz; i++ ){
			tOut [i] = float(tOut_half[i]);

}

    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);
    return;
}
