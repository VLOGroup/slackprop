// This file is part of slack-prop.
//
// Copyright (C) 2017 Sasha Shekhovtsov <shekhovtsov at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// dvs-panotracking is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// slack-prop is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <cuda.h>
#include <device_functions.h>
#include <math_constants.h>
#include <stdint.h>

#include "slp_util.h"

//#ifndef slack_prop_util_cuh
//#define slack_prop_util_cuh

//#include "ndarray/error.h"
//#include "ndarray/ndarray_ref.h"

//#include <cuda.h>
//#include <device_functions.h>
//#include <device_functions.hpp>

#ifndef  __CUDA_ARCH__
//#error This File should not be included in cpp mode
#endif

#define FINF CUDART_INF_F


// functions
//_______________________________________________________________________________________________
//_______________________________________________________________________________________________

namespace device{
	//! non-volatile min
	__device__ __forceinline__ float min(float x, float y){
		float ret;
		asm("min.f32 %0, %1, %2;" : "=f"(ret) : "f"(x), "f"(y));
		return ret;
	}

	__device__ __inline__ uint32_t laneid(){
	  uint32_t laneid;
	  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	  return laneid;
	}

	__device__ __inline__ uint32_t warpid(){
	  uint32_t warpid;
	  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
	  return warpid;
	}

	__device__ __inline__ uint32_t nwarpid(){
		uint32_t nwarpid;
		asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(nwarpid));
		return nwarpid;
	}
};


template<bool up = true>
static __device__ __forceinline__ float shfl_up_down(const float & val, int delta, const int width) {
	//! differs from __shfl_up / __shfl_down in  NOT volatile asm (volatile prevents const load optimization and reordering)
	// note: shfl_up reads from  lane - bval (from thread before)
	// note: shfl_dwn reads from lane + bval (from thread after)
	float ret;
	if (up){
		const int c = (WARP_SIZE - width) << 8;
		asm("shfl.up.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(val), "r"(delta), "r"(c));
	} else{
		const int c = ((WARP_SIZE - width) << 8) | 0x1f;
		asm("shfl.down.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(val), "r"(delta), "r"(c));
	};
	return ret;
}


template<bool up = true>
static __device__ __forceinline__ unsigned int shfl_up_down(const unsigned int & val, int delta, const int width) {
	//! differs from __shfl_up / __shfl_down in  NOT volatile asm (volatile prevents const load optimization and reordering)
	// note: shfl_up reads from  lane - bval (from thread before)
	// note: shfl_dwn reads from lane + bval (from thread after)
	unsigned int ret;
	if (up){
		const int c = (WARP_SIZE - width) << 8;
		asm("shfl.up.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(delta), "r"(c));
	} else{
		const int c = ((WARP_SIZE - width) << 8) | 0x1f;
		asm("shfl.down.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(delta), "r"(c));
	};
	return ret;
}

template<typename type>
__device__ __forceinline__ type shfl_up(const type & val, int delta, const int width) {
	//static_assert(sizeof(type) == sizeof(unsigned int),"bad");
	//return (const type&&)shfl_up_down<true>((const float&)val, delta, width);
	//return (const type&&)shfl_up_down<true>((const unsigned int&)val, delta, width);
	return shfl_up_down<true>(val, delta, width);
}

template<typename type>
__device__ __forceinline__ type shfl_down(const type & val, int delta, const int width) {
	//static_assert(sizeof(type) == sizeof(float),"bad");
	//return (const type&&)shfl_up_down<false>((const float&)val, delta, width);
	//static_assert(sizeof(type) == sizeof(unsigned int),"bad");
	//return (const type&&)shfl_up_down<false>((const unsigned int&)val, delta, width);
	return shfl_up_down<false>(val, delta, width);
}

template<bool up = true>
__device__ __forceinline__ float shfl_up_down(const float & val, const float & val_out_bound, const int delta, const int width) {
	//! differs from __shfl_up in using predicate to set out of bounds value and NOT volatile asm (volatile prevents const load optimization and reordering)
	float ret;
	if (up){
		const int c = (WARP_SIZE - width) << 8;
		asm("{\n\t"
			" .reg .pred p;\n\t"
			" shfl.up.b32 %0|p, %1, %2, %3;\n\t"
			" @!p mov.f32 %0, %4;\n\t"
			"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
	} else{
		const int c = ((WARP_SIZE - width) << 8) | 0x1f;
		asm("{\n\t"
			" .reg .pred p;\n\t" //
			" shfl.down.b32 %0|p, %1, %2, %3;\n\t" //
			" @!p mov.f32 %0, %4;\n\t" //"
			"}" : "=f"(ret) : "f"(val), "r"(delta), "r"(c), "f"(val_out_bound));
	};
	return ret;
}

template<typename type>
__device__ __forceinline__ type shfl_up(const type & val, const type & val_out_bound, const int delta, const int width) {
	//return (const type&)shfl_up_down<true>((const float&)val,val_out_bound,delta,width);
	return shfl_up_down<true>(val,val_out_bound,delta,width);
}

template<typename type>
__device__ __forceinline__ float shfl_down(const type & val, const type & val_out_bound, const int delta, const int width) {
	//return (const type&)shfl_up_down<false>((const float&)val, (const float&)val_out_bound, delta, width);
	return shfl_up_down<false>(val, val_out_bound, delta, width);
}

__device__ __forceinline__ float shfl_xor_c(float var, int laneMask, const int c) {
	float ret;
	//const int c = global.shfl_dwn_c;
	//c = ((WARP_SIZE - width) << 8) | 0x1f;
	asm ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(c));
	return ret;
}

__device__ __forceinline__ float group_min(float val, const int width){

	// reference solution
	/*
	volatile __shared__ float vals[2048];
	int id = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
	vals[id] = val;
	int my_group = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x;
	float ref_min = FINF;
	for(int i=0; i<LABELS_GROUP; ++i){
		ref_min = device::min(ref_min,vals[my_group+i]);
	};
	*/
	//int c = global.shfl_dwn_c;
	const int c = ((WARP_SIZE - width) << 8) | 0x1f;
	if (width > 16)val = device::min(val, shfl_xor_c(val, 16, c)); // reduce full warp
	if (width >  8)val = device::min(val, shfl_xor_c(val,  8, c)); // or any smaller power of 2
	if (width >  4)val = device::min(val, shfl_xor_c(val,  4, c)); //
	if (width >  2)val = device::min(val, shfl_xor_c(val,  2, c));
	if (width >  1)val = device::min(val, shfl_xor_c(val,  1, c));

	return val;
}

__device__ __forceinline__ float group_sum(float val, const int width){
	//int c = global.shfl_dwn_c;
	const int c = ((WARP_SIZE - width) << 8) | 0x1f;
	if (width > 16)val += shfl_xor_c(val, 16, c); // reduce full warp
	if (width >  8)val += shfl_xor_c(val,  8, c); // or any smaller power of 2
	if (width >  4)val += shfl_xor_c(val,  4, c); //
	if (width >  2)val += shfl_xor_c(val,  2, c);
	if (width >  1)val += shfl_xor_c(val,  1, c);
	return val;
}

template<bool forward = true, bool inclusive = true>
__device__ __forceinline__ float group_cum_min(float val, const int width, const int cutoff){
	//! cutoff must be less equal width, this is limit beyond which cumulative min
	// does not propagate - saves few instructions for small width kernels
	// width is not constexpr so expand explicitly
	if (!inclusive){
		// exclusive prefix min, first reads all values from the left
		val = shfl_up_down<forward>(val, FINF, 1, width); // load INF for out of bounds thread
	};
	// now compute inclusive prefix min
	// min is idempotent, predicating out out-of-bound values not needed
	if (cutoff >  1)val = device::min(val, shfl_up_down<forward>(val,  1, width));
	if (cutoff >  2)val = device::min(val, shfl_up_down<forward>(val,  2, width));
	if (cutoff >  4)val = device::min(val, shfl_up_down<forward>(val,  4, width));
	if (cutoff >  8)val = device::min(val, shfl_up_down<forward>(val,  8, width));
	if (cutoff > 16)val = device::min(val, shfl_up_down<forward>(val, 16, width));
	return val;
}

//! doing up and down in one loop decreases data dependency (compiler optimizer lacks to introduce registers)
template<bool inclusive = true>
__device__ __forceinline__ void group_cum_min_up_down(float & val_up, float & val_dwn, const int width, const int cutoff){
	//! cutoff must be less equal width, this is limit beyond which cumulative min
	// does not propagate - saves few instructions for small width kernels
	// width is not constexpr so expand explicitly
	if (!inclusive){
		// exclusive prefix min, first reads all values from the left
		val_up = shfl_up(val_up, FINF, 1, width); // load INF for out of bounds thread
		val_dwn = shfl_down(val_dwn, FINF, 1, width); // load INF for out of bounds thread
	};
	// now compute inclusive prefix min
	// min is idempotent, predicating out out-of-bound values not needed
	for(int i=1; i <= 16; i <<= 1){
		if (cutoff >  i){
			const float v1 = shfl_up(val_up,  i, width);
			const float v2 = shfl_down(val_dwn, i, width);
			val_up = device::min(val_up, v1);
			val_dwn = device::min(val_dwn, v2);
		};
	};
}
