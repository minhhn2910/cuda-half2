#include <cuda_fp16.h>
//#define __hdiv hdiv
#define __hexp hexp

__device__ inline __half habsf(__half d)
{
	(*((unsigned short*)(&d))) &= 0x7fff; //set 2 sign bits to 0
	return d;
}
/*__device__ inline __half h2abs(__half d)
{
//	(*((unsigned int*)(&d))) &= 0x7fff7fff; //set 2 sign bits to 0
	float2 value = __half2float2(d);
	value.x = fabsf(value.x);
	value.y = fabsf(value.y);
	
	return __float22half_rn(value);
}
*/
__device__ inline __half hnegf(__half d)
{
	(*((unsigned short*)(&d))) ^= 0x8000; //flip 2 sign bits
	return d;
}

/*
__device__ inline __half h2neg(__half d)
{
	float2 value = __half2float2(d);
	value.x = 0 - value.x;
	value.y = 0 - value.y;
	
	return __float22half_rn(value);
}
*/
/*
inline __device__ __half operator + (const __half a, const __half b) {
   return __hadd(a, b);
 }
inline __device__ __half operator * (const __half a, const __half b) {
   return __hmul(a, b);
 }
inline __device__ __half operator - (const __half a, const __half b) {
   return __hsub(a, b);
 }
inline __device__ __half operator / (const __half a, const __half b) {
	return __hdiv(a,b);
 }
inline __device__ __half operator - (const __half a) {
   return hnegf(a);
 }
inline __device__ __half& operator += (__half& a, const __half b) {
   a = a + b;
   return a;
 }
inline __device__ __half& operator *= (__half& a, const __half b) {
   a = a * b;
   return a;
 }
inline __device__ __half& operator -= (__half& a, const __half b) {
   a = a - b;
   return a;
 }
inline __device__ __half& operator /= (__half& a, const __half b) {
   a = a / b;
   return a;
 }
//~ inline __device__ __half& operator = (__half& a, const __half b) {
   //~ a =  __float2half(b);
   //~ return a;
 //~ }

inline __device__ bool operator == (const __half a, const __half b) {
   return __heq(a, b);
 }
inline __device__ bool operator != (const __half a, const __half b) {
   return __hne(a, b);
 }
inline __device__ bool operator < (const __half a, const __half b) {
   return __hlt(a, b);
 }
inline __device__ bool operator <= (const __half a, const __half b) {
   return __hle(a, b);
 }
inline __device__ bool operator > (const __half a, const __half b) {
   return __hgt(a, b);
 }
inline __device__ bool operator >= (const __half a, const __half b) {
   return __hge(a, b);
 }

*/

inline __device__ __half operator + (const float a, const __half b) {
   return __hadd(__float2half(a), b);
 }
inline __device__ __half operator * (const float a, const __half b) {
   return __hmul(__float2half(a), b);
 }
inline __device__ __half operator - (const float a, const __half b) {
   return __hsub(__float2half(a), b);
 }
inline __device__ __half operator / (const float a, const __half b) {
	return __hdiv(__float2half(a),b);
 }
inline __device__ __half operator + (const __half a, const float b) {
   return __hadd(a, __float2half(b));
 }
inline __device__ __half operator * (const __half a, const float b) {
   return __hmul(a, __float2half(b));
 }
inline __device__ __half operator - (const __half a, const float b) {
   return __hsub(a, __float2half(b));
 }
inline __device__ __half operator / (const __half a, const float b) {
	return __hdiv(a,__float2half(b));
 }


//mathfunc float
inline __device__ __half __fdividef(const __half a, const __half b){
	return a/b;
}
//mathfunc float
inline __device__ __half __fdividef(const float a, const __half b){
	return __float2half(a)/b;
}
//mathfunc float
inline __device__ __half __fdividef(const __half a, const float b){
	return a/__float2half(b);
}
inline __device__ __half cosf(const __half a){
	return hcos(a);
}
inline __device__ __half sinf(const __half a){
	return hsin(a);
	}
inline __device__ __half expf(const __half a){
	return hexp(a);
	}
inline __device__ __half __expf(const __half a){
	return hexp(a);
	}
//~ #define FLOAT_EXP10 "exp10f("
//~ #define FLOAT_EXP2 "exp2f("
inline __device__ __half logf(const __half a){
	return hlog(a);
	}
inline __device__ __half __logf(const __half a){
	return hlog(a);
	}
//~ #define FLOAT_LOG10 "log10f("
//~ #define FLOAT_LOG2 "log2f("
inline __device__ __half rsqrtf(const __half a){
	return hrsqrt(a);
	}
inline __device__ __half sqrtf(const __half a){
	return hsqrt(a);
	}
inline __device__ __half fabsf(const __half a){
	return habsf(a);
	}



// math funcs
//~ #define HALF2_COS "hcos("
//~ #define HALF2_SIN "hsin("
//~ #define HALF2_EXP "hexp("
//~ #define HALF2_EXP10 "hexp10("
//~ #define HALF2_EXP2 "hexp2("
//~ #define HALF2_LOG "hlog("
//~ #define HALF2_LOG10 "hlog10("
//~ #define HALF2_LOG2 "hlog2("
//~ #define HALF2_RSQRT "hrsqrt("
//~ #define HALF2_SQRT "hsqrt("
//~ #define HALF2_ABS "habs("
