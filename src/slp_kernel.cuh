#pragma once

/*
#ifndef DEBUG // RELEASE
	#define DEBUG_LVL -1
	#define RINLINE __forceinline__
#else // DEBUG
	#define DEBUG_LVL -1
	#define RINLINE __noinline__
#endif
*/

#ifdef NDEBUG // Release mode
	#define RINLINE __forceinline__
#else
	#define RINLINE
#endif

//#define RINLINE __forceinline__

#define DEBUG_LVL -1


#include <cuda.h>
#include <assert.h>
#include "slp_util.cuh"
#include "slp_kernel.h"
#include <math_constants.h>
//#include <device_launch_parameters.h>
//#include <device_functions.hpp>
//#include <device_functions_decls.h>

//#define FINF CUDART_INF_F
#define MAX_BLOCK_WARPS 8

#define def_max(x1,x2) ( (x1>=x2) ? (x1) : (x2) )
#define def_abs(x)  ((x)>=0 ? (x) : -(x))
#define def_abs_dif(x1,x2)  ((x1>=x2) ? (x1-(x2)) : (x2-(x1)) )

/*
namespace device{
	__device__ __forceinline__ float min(float x, float y){
		float ret;
		asm("min.f32 %0, %1, %2;" : "=f"(ret) : "f"(x), "f"(y));
		return ret;
	};
};
*/

using namespace device;

//______________________________________________
//__constant__ __device__ int M_lookup[MAX_CHAIN_LENGTH];
const __constant__ __device__ global_info_cmem global;

//int * get_M_lookup_dev_ptr();
global_info_cmem * get_global_dev_ptr();

//________________segment_info___________________________________________
class segment_info_cmem_d : public segment_info_cmem{ // info per varied x offset
public:
	__device__ __forceinline__ dslice<warp_packf,1> C_in_slice(int dx, int line_num)const{
		int x = x1 + dx*dir();
		return dslice<warp_packf, 1>((global.C_in.ptr(x, line_num) + threadIdx.x), global.C_in.stride<char>(0).value() * dir());
	}

	__device__ __forceinline__ tstride<char> C_stride0()const{
		//return global.C_in.stride[0] * dir();
		return global.C_in.stride<char>(0) * dir();
	}

	__device__ __forceinline__ tstride<char> C_stride1()const{
		//return global.C_in.stride[1];
		return global.C_in.stride<char>(1);
	}

	/*
	__device__ __forceinline__ int C_stride0()const{
		return global.C_in.stride(0) * dir();
	};
	__device__ __forceinline__ int C_stride1()const{
		return global.C_in.stride(1);
	};
	*/
	__device__ __forceinline__ const warp_packf & C_in(int dx, int line_num)const{ // should act as a ndarray_ref
		int x = x1 + dx*dir();
		return *(global.C_in.ptr(x, line_num) + threadIdx.x);
	}

	__device__ __forceinline__ warp_packf & C_out(int dx, int line_num){
		int x = x1 + dx*dir();
		return *(global.C_out.ptr(x, line_num) + threadIdx.x);
		//return *(global.C_out.begin() + (x1 + dx)*C_stride0() + line_num*C_stride1());
	}

	__device__ __forceinline__ warp_packf & C_total(int dx, int line_num)const{
		int x = x1 + dx*dir();
		return *(global.C_total.ptr(x, line_num) + threadIdx.x);
		//int x = x1 + dx*dir();
		//assert(x >=0 && x < global.C_total.sz[0]);
		//assert(line_num >=0 && line_num < global.C_total.sz[1]);
		//return *(global.C_total.begin() + (x1 + dx)*C_stride0() + line_num*C_stride1());
	}

	__device__ __forceinline__ warp_packf & M_in(int dx, int line_num)const{
		int x = x1 + dx*dir();
		return *(global.M_in.ptr(x, line_num) + threadIdx.x);
	}

	__device__ __forceinline__ int & x_out(int dx, int line_num)const{
		int x = x1 + dx*dir();
		return global.x_out(x, line_num);
	}

	__device__ __forceinline__ float & LB_out(int dx, int line_num)const{
		int x = x1 + dx*dir();
		return global.LB_out(x, line_num);
	}

	__device__ __forceinline__ float * weights(int dx, int line_num)const{
		int x = x1 + dx*dir();
		if (dir() < 1){
			x = x - 1; // right to left weight is at the left pixel
		};
		//return global.weights(x, line_num);
		return global.weights.ptr(x, line_num);
	}

	__device__ __forceinline__ tstride<char> weights_stride0()const{
		return global.weights.stride<char>(0) * dir();
	}

	/*
	__device__ __forceinline__ int weights_stride0()const{
		return global.weights.stride(0) * dir();
	};
	*/
};
//_______________________________________________________________
//extern __constant__ __device__ segment_info_cmem_d cmem_segments[MAX_SEGMENTS];
segment_info_cmem * get_segments_dev_ptr();
//______________________________________________

//template<bool up = true>
//static __device__ __forceinline__ float shfl_up(const float & val, int delta, const int width) {
//	//! differs from __shfl_up / __shfl_down in  NOT volatile asm (volatile prevents const load optimization and reordering)
//	// note: shfl_up reads from  lane - bval (from thread before)
//	// note: shfl_dwn reads from lane + bval (from thread after)
//	float ret;
//	if (up){
//		const int c = (warpSize - width) << 8;
//		asm("shfl.up.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(val), "r"(delta), "r"(c));
//	} else{
//		const int c = ((warpSize - width) << 8) | 0x1f;
//		asm("shfl.down.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(val), "r"(delta), "r"(c));
//	};
//	return ret;
//};
//
//static __device__ __forceinline__ float shfl_up(const float & val, int delta, const int width) {
//	return shfl_up<true>(val, delta, width);
//};
//
//static __device__ __forceinline__ float shfl_down(const float & val, int delta, const int width) {
//	return shfl_up<false>(val, delta, width);
//};
//
//template<bool up = true>
//static __device__ __forceinline__ float shfl_up(const float & val, const float & val_out_bound, const int delta, const int width) {
//	//! differs from __shfl_up in using predicate to set out of bounds value and NOT volatile asm (volatile prevents const load optimization and reordering)
//	float ret;
//	if (up){
//		const int c = (warpSize - width) << 8;
//		asm("{\n\t"
//			" .reg .pred p;\n\t"
//			" shfl.up.b32 %0|p, %1, %2, %3;\n\t"
//			" @!p mov.f32 %0, %4;\n\t"
//			"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
//	} else{
//		const int c = ((warpSize - width) << 8) | 0x1f;
//		asm("{\n\t"
//			" .reg .pred p;\n\t" //
//			" shfl.down.b32 %0|p, %1, %2, %3;\n\t" //
//			" @!p mov.f32 %0, %4;\n\t" //"
//			"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
//	};
//	return ret;
//};
//
//static __device__ __forceinline__ float shfl_up(const float & val, const float & val_out_bound, const int delta, const int width) {
//	/*
//	//! differs from __shfl_up in using predicate to set out of bounds value and NOT volatile asm (volatile prevents const load optimization and reordering)
//	float ret;
//	//const int c = global.shfl_up_c;
//	const int c = (warpSize - width) << 8;
//	asm ("{\n\t"
//		" .reg .pred p;\n\t"
//		" shfl.up.b32 %0|p, %1, %2, %3;\n\t"
//		" @!p mov.f32 %0, %4;\n\t"
//		"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
//	return ret;
//	*/
//	return shfl_up<true>(val,val_out_bound,delta,width);
//};
//
//static __device__ __forceinline__ float shfl_down(const float & val, const float & val_out_bound, const int delta, const int width) {
//	/*
//	float ret;
//	//const int c = global.shfl_dwn_c;
//	const int c = ((warpSize - width) << 8) | 0x1f;
//	asm ("{\n\t"
//		" .reg .pred p;\n\t" //
//		" shfl.down.b32 %0|p, %1, %2, %3;\n\t" //
//		" @!p mov.f32 %0, %4;\n\t" //"
//		"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
//	return ret;
//	*/
//	return shfl_up<false>(val, val_out_bound, delta, width);
//};
//
//static __device__ __forceinline__ float shfl_xor_c(float var, int laneMask, const int c) {
//	float ret;
//	//const int c = global.shfl_dwn_c;
//	//c = ((warpSize - width) << 8) | 0x1f;
//	asm ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(c));
//	return ret;
//}
//
//static __device__ __forceinline__ float group_min(float val){
//
//	// reference solution
//	/*
//	volatile __shared__ float vals[2048];
//	int id = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
//	vals[id] = val;
//	int my_group = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x;
//	float ref_min = FINF;
//	for(int i=0; i<LABELS_GROUP; ++i){
//		ref_min = device::min(ref_min,vals[my_group+i]);
//	};
//	*/
//
//	//int c = global.shfl_dwn_c;
//	const int width = LABELS_GROUP;
//	const int c = ((WARP_SIZE - width) << 8) | 0x1f;
//	if (width > 16)val = device::min(val, shfl_xor_c(val, 16, c)); // reduce full warp
//	if (width >  8)val = device::min(val, shfl_xor_c(val,  8, c)); // or any smaller power of 2
//	if (width >  4)val = device::min(val, shfl_xor_c(val,  4, c)); //
//	if (width >  2)val = device::min(val, shfl_xor_c(val,  2, c));
//	if (width >  1)val = device::min(val, shfl_xor_c(val,  1, c));
//
//	// second reference sol
//	/*
//	#pragma unroll
//	for (int i = 1; i < LABELS_GROUP; i = i << 1){
//		val = device::min(val, __shfl_xor(val, i, LABELS_GROUP));
//	};
//	 */
//	//if(val != ref_min) val = FINF;
//	//val = ref_min;
//
//	return val;
//};
//
//template<bool forward = true, bool inclusive = true>
//static __device__ __forceinline__ float group_cum_min(float val, const int width, const int cutoff){
//	//! cutoff must be less equal width, this is limit beyond which cumulative min
//	// does not propagate - saves few instructions for small width kernels
//	// width is not constexpr so expand explicitly
//	if (!inclusive){
//		// exclusive prefix min, first reads all values from the left
//		val = shfl_up<forward>(val, FINF, 1, width); // load INF for out of bounds thread
//	};
//	// now compute inclusive prefix min
//	// min is idempotent, predicating out out-of-bound values not needed
//	if (cutoff >  1)val = device::min(val, shfl_up<forward>(val,  1, width));
//	if (cutoff >  2)val = device::min(val, shfl_up<forward>(val,  2, width));
//	if (cutoff >  4)val = device::min(val, shfl_up<forward>(val,  4, width));
//	if (cutoff >  8)val = device::min(val, shfl_up<forward>(val,  8, width));
//	if (cutoff > 16)val = device::min(val, shfl_up<forward>(val, 16, width));
//	return val;
//};
//
////! doing up and down in one loop decreases data dependency (compiler optimizer lacks to introduce registers)
//template<bool inclusive = true>
//static __device__ __forceinline__ void group_cum_min_up_down(float & val_up, float & val_dwn, const int width, const int cutoff){
//	//! cutoff must be less equal width, this is limit beyond which cumulative min
//	// does not propagate - saves few instructions for small width kernels
//	// width is not constexpr so expand explicitly
//	if (!inclusive){
//		// exclusive prefix min, first reads all values from the left
//		val_up = shfl_up(val_up, FINF, 1, width); // load INF for out of bounds thread
//		val_dwn = shfl_down(val_dwn, FINF, 1, width); // load INF for out of bounds thread
//	};
//	// now compute inclusive prefix min
//	// min is idempotent, predicating out out-of-bound values not needed
//	//const int c_up  = (warpSize - width) << 8;
//	//const int c_dwn = ((warpSize - width) << 8) | 0x1f;
//
//	for(int i=1; i <= 16; i <<= 1){
//		if (cutoff >  i){
//			//val_up = device::min(val_up, shfl_up(val_up,  i, width));
//			//val_dwn = device::min(val_dwn, shfl_down(val_dwn,  i, width));
//			//const float v2 = shfl_down(val_dwn, i, width);
//			/*
//			float v1 = val_up;
//			float v2 = val_dwn;
//			//					" .reg .pred p1;\n\t" //
//			//					" .reg .pred p2;\n\t" //
//			//					" .reg .f32 v1;\n\t" //
//			//					" .reg .f32 v2;\n\t" //
//			asm("{\n\t"
//					" shfl.up.b32 %0, %0, %2, %3;\n\t" //
//					" shfl.down.b32 %1, %1, %2, %4;\n\t" //
//					"}" : "+f"(val_up) , "+f"(val_dwn) : "r"(i), "r"(c_up), "r"(c_dwn));
//			//const float v1 = shfl_up(val_up,  i, width);
//			//const float v2 = shfl_down(val_dwn, i, width);
//			val_up = device::min(val_up, v1);
//			val_dwn = device::min(val_dwn, v2);
//			*/
//			const float v1 = shfl_up(val_up,  i, width);
//			const float v2 = shfl_down(val_dwn, i, width);
//			val_up = device::min(val_up, v1);
//			val_dwn = device::min(val_dwn, v2);
//		};
//	};
//};


__device__ void print_pack(const char * name, const warp_packf & a, int debug_lvl = 1);

__device__ void print_pack(const char * name, const float & a, int debug_lvl = 1);

__device__ __forceinline__ warp_packf pw_term_device::message(const warp_packf & cm, const float * __restrict__ W, bool dir){
	if (delta == 2){
		return message_L1L2(cm, W, dir);
	} else{
		return message_TTV(cm, W, dir);
	};
};

__device__ RINLINE warp_packf pw_term_device::message_L1L2(const warp_packf & cm, const float * __restrict__ W, bool dir){
	warp_packf m;
	// fetch parameters
	const float L1 = W[0];
	//const float L2 = W[1];
	const float L2 = L1 + W[1]; //assume delta = 2, W[1] is a slope
	// pass message
	//float full_min = cm.min();
	//print_pack("cm", cm, 2);
	float c_above = shfl_down(cm[0], cm[LABELS_PACK - 2], 1, LABELS_GROUP); // 1
	float c_below = shfl_up(cm[LABELS_PACK - 1], cm[1], 1, LABELS_GROUP); // 1
	if (threadIdx.x == 0){
		//print_pack("below", c_below, 3);
	};
	for (int i = 1; i < LABELS_PACK - 1; ++i){
		m[i] = device::min(cm[i - 1], cm[i + 1]);
	};
	//float full_min = min(m[1], m[2]); //todo: expand // 1
	m[0] = device::min(c_below, cm[1]);
	m[LABELS_PACK - 1] = device::min(c_above, cm[LABELS_PACK - 2]);
	//reduce rest of a down to full min, now full_min keeps min of LABELS_PACK*LABELS_GROUP elements
	//print_pack("cm",cm,1);
	float full_min = group_min(cm.min(),LABELS_GROUP);
	// now every thread holds full_min for its line
	m += L1;
	full_min += L2;
	m = min(m, cm);
	// unnormalized:
	m = min(m, full_min);
	// normalized:
	//m = min(m - full_min, 0.0f);
	return m;
};

__device__ RINLINE warp_packf pw_term_device::message_TTV(const warp_packf & cm, const float * __restrict__ W, bool dir){
	//
	// TODO: shorten cascade length by doing pack/2 in parallel (ILP), full min through SM
	// fetch parameters
	const float L1 = W[0];
	const float slope = W[1];
	//const float L2 = W[1];
	//const float L1 = this->L1*weight;
	//const float L2 = this->L2*weight;
	const int cutoff = this->cutoff;
	//
	warp_packf m; // output message
	// temps
	float l_scan;
	float r_scan;
	const float gramp = slope*(threadIdx.x*LABELS_PACK);
	// forward (prefix) pack scan and pack min
	float full_min = group_min(cm.min(),LABELS_GROUP); // robust term
	//
	l_scan = cm[0] + L1 - (0+1)*slope; // prefix scan
	m[0] = cm[0];
	for (int i = 1; i < LABELS_PACK; ++i){
		m[i] = device::min(cm[i], l_scan + i*slope); // envelope inside pack
		l_scan = device::min(l_scan, cm[i] + L1 - (i+1)*slope); // prefix within pack
	};
	//l_scan = group_cum_min<true, false>(l_scan, LABELS_GROUP, cutoff); // exclusive prefix scan within each group (pixel)
	//
	// backward pack scan
	r_scan = cm[LABELS_PACK - 1] + L1 + ((LABELS_PACK-1)-1)*slope;
	for (int i = LABELS_PACK - 2; i >= 0; --i){
		m[i] = device::min(m[i], r_scan - i*slope); // envelope inside pack
		r_scan = device::min(r_scan, cm[i] + L1 + (i-1)*slope); // postfix within pack
	};
	l_scan -= gramp;
	r_scan += gramp;
	group_cum_min_up_down<false>(l_scan, r_scan, LABELS_GROUP, cutoff); // exclusive prefix scan within each group (pixel)
	//
 	//r_scan = group_cum_min<false, false>(r_scan, LABELS_GROUP, cutoff);// exclusive postfix scan
	//l_scan = group_cum_min<true, false>(l_scan, LABELS_GROUP, cutoff); // exclusive prefix scan within each group (pixel)

	r_scan -= gramp; // also relative to pack begin
	l_scan += gramp; // relative to pack begin
	// combine together
	float L2 = L1 + (delta - 1)*slope;
	for (int i = 0; i < LABELS_PACK; ++i){
		m[i] = device::min(m[i], full_min + L2); // robust part
		m[i] = device::min(m[i], l_scan + i*slope);
		m[i] = device::min(m[i], r_scan - i*slope);
	};
	return m;
};

//__device__ __forceinline__ warp_packf pw_term_device::message(const warp_packf & cm1, const float weight){
//	warp_packf m;
//	// pass message
//	float full_min = group_min(cm1.min());
//	//
//	warp_packf cm = cm1 - full_min; // normalize
//	float c_above = shfl_down(cm[0], cm[LABELS_PACK - 2], 1, LABELS_GROUP); // 1
//	float c_below = shfl_up(cm[LABELS_PACK - 1], cm[1], 1, LABELS_GROUP); // 1
//	if (threadIdx.x == 0){
//		print_pack("below", c_below, 3);
//	};
//	for (int i = 1; i < LABELS_PACK - 1; ++i){
//		m[i] = device::min(cm[i - 1], cm[i + 1]);
//	};
//	m[0] = device::min(c_below, cm[1]);
//	m[LABELS_PACK - 1] = device::min(c_above, cm[LABELS_PACK - 2]);
//	m += L1*weight;
//	full_min = L2*weight;
//	m = min(m, cm);
//	m = min(m, full_min);
//	return m;
//};

__device__ void RINLINE pw_term_device::handshake(const warp_packf & c1, warp_packf & cm1, const warp_packf & c2, warp_packf & cm2, const float coeff, const float * __restrict__ W, bool dir){
	// pass from cm1 to cm2
	warp_packf m1 = message(cm1, W, dir);
	warp_packf M = m1 + cm2; // full min-marginal
	//print_pack("M",M,2);
	/*
	if (DEBUG_LVL >= 1){
		float minF = group_min(M.min());
		if (threadIdx.x == 0){
			printf("handshake: minM[b%i]=%f\n", blockDim.z*blockIdx.z + threadIdx.z, minF);
		};
	};
	*/
	//print_pack("M",M,3);
	// divide by 2
	//M *= 0.5;
	M *= coeff;
	// share to cm1
	warp_packf m = message(M - m1, W, dir);
	cm1 = c1 + m;
	// bounce back
	cm2 = c2 + message(-m, W, dir);
};
