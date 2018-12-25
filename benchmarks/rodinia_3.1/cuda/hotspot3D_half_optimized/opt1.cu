long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#include "newhalf.hpp"
#include <cuda_fp16.h>
#include "half_operator_overload.cuh"
__global__ void hotspotOpt1(__half *p, __half* tIn, __half *tOut, __half sdc,
        int nx, int ny, int nz,
        __half ce_t, __half cw_t,
        __half cn_t, __half cs_t,
        __half ct_t, __half cb_t,
        __half cc_t)
{

	__half ce = ce_t; __half cw = cw_t;
	__half cn = cn_t; __half cs = cs_t;
	__half ct = ct_t; __half cb = cb_t;
	__half cc = cc_t;
    __half amb_temp = __float2half(80.0);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
		//add in for performance measurement
#pragma unroll 1
	for(int run = 0; run <100; run++){
    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    __half temp1, temp2, temp3;

		__half tinw, tine, tins, tinn, pc;

		tinw = tIn[W];
		tine = tIn[E];
		tins =  tIn[S];
		tinn = tIn[N] ;
		pc = p[c];

    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tinw + ce * tine + cs * tins
        + cn * tinn + cb * temp1 + ct * temp3 + sdc * pc + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
		#pragma unroll 1
    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];

				tinw = tIn[W];
				tine = tIn[E];
				tins =  tIn[S];
				tinn = tIn[N] ;
				pc = p[c];
				tOut[c] = cc * temp2 + cw * tinw + ce * tine + cs * tins
		        + cn * tinn + cb * temp1 + ct * temp3 + sdc * pc + ct * amb_temp;
    //    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    //        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;

		tinw = tIn[W];
		tine = tIn[E];
		tins =  tIn[S];
		tinn = tIn[N] ;
		pc = p[c];
		tOut[c] = cc * temp2 + cw * tinw + ce * tine + cs * tins
				+ cn * tinn + cb * temp1 + ct * temp3 + sdc * pc + ct * amb_temp;
  //  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
  //      + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;

} //endof loop

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



    __half  *tIn_d, *tOut_d, *p_d;

		printf ("val: %f %f %f %f %f %f %f %f \n",stepDivCap,ce,cw,cn,ct,cb,cc);
    cudaMalloc((void**)&p_d,s);
    cudaMalloc((void**)&tIn_d,s);
    cudaMalloc((void**)&tOut_d,s);
    cudaMemcpy(tIn_d, tIn_half, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p_half, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

		half_float::half stepDivCap_half = half_float::half(stepDivCap);
		__half stdc = *(__half*)&(stepDivCap_half);

		half_float::half ce_half = half_float::half(ce);
		__half ce_dev = *(__half*)&ce_half;

		half_float::half cw_half = half_float::half(cw);
		__half cw_dev = *(__half*)&cw_half;

		half_float::half cn_half = half_float::half(cn);
		__half cn_dev = *(__half*)&cn_half;

		half_float::half cs_half = half_float::half(cs);
		__half cs_dev = *(__half*)&cs_half;

		half_float::half ct_half = half_float::half(ct);
		__half ct_dev = *(__half*)&ct_half;

		half_float::half cb_half = half_float::half(cb);
		__half cb_dev = *(__half*)&cb_half;

		half_float::half cc_half = half_float::half(cc);
		__half cc_dev = *(__half*)&cc_half;

    long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (p_d, tIn_d, tOut_d, stdc, nx, ny, nz, ce_dev, cw_dev, cn_dev, cs_dev, ct_dev, cb_dev, cc_dev);
        __half *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
    cudaDeviceSynchronize();
    long long stop = get_time();
    float time = (float)((stop - start)/(1000.0 * 1000.0));
    printf("Time: %.3f (s)\n",time);
    cudaMemcpy(tOut_half, tOut_d, s, cudaMemcpyDeviceToHost);

		for(int i = 0; i < nx * ny * nz; i++ ){
			tOut [i] = float(tOut_half[i]);

}

    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);
    return;
}
