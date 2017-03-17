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

#define WARP_SIZE 32
#define divup(x,y) ((x-1)/(y)+1)
#define roundup(x,y) (divup(x,y)*y)
#define rounddown(x,y) ((x/y)*y)

#ifndef  __CUDA_ARCH__
using namespace std;
#endif

// compile-time 2^x and log_2 x

//! compile time 2^N
constexpr int c2pow(int N){
	return (N>0) ? 2 * c2pow(N - 1) : 1;
}

//! compile time log_2, integer, rounded down
constexpr int clog2(int N){
	return (N>1) ? 1 + clog2(N / 2) : 0;
}

//! compile time log_2, integer, rounded up
constexpr int log2up(int N){
	return (N>1) ? 1 + log2up(divup(N, 2) ) : 0;
}

constexpr int cpow_up(int N){
	return c2pow(log2up(N));
}

template<typename type>
constexpr int cmax(type a, type b){
	return a > b? a : b;
}

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))

/*
template <int N> inline constexpr int t2pow(){
	return (N>0) ? 2 * t2pow<N - 1>() : 1;
}
template <> inline constexpr int t2pow<0>(){
	return 1;
}

template <int N> inline const int tlog2(){
	return (N>1) ? 1 + tlog2<N / 2>() : 0;
}
template <> inline const int tlog2<1>(){
	return 0;
}
*/

template <typename type, int size>
struct small_array{ // __align__(16)
public:
	type a[size];
	__device__ __forceinline__ type & operator[](int i){ return a[i]; };
	__host__ __device__ __forceinline__ const type & operator[](int i)const{ return a[i]; };
	__host__ __device__ __forceinline__ small_array(){
		static_assert(sizeof(type)*size == sizeof(small_array<type,size>),"alignment requirement");
	};
	/*
	__device__ __forceinline__ small_array(const small_array<type, size> & b){
		operator = (b);
	};

	__device__ __forceinline__ void operator = (const small_array<type, size> & b){
		int i = 0;
#pragma unroll
		for (; i <= size - 4; i += 4){ // bulk copy in 16 bytes
			*(float4*)&a[i] = *(float4*)&b.a[i];
		};
#pragma unroll
		for (; i <= size - 2; i += 2){ // smaller copy: 8 bytes
			*(float2*)&a[i] = *(float2*)&b.a[i];
		};
#pragma unroll
		for (; i <= size - 1; ++i){ // smallest copy 4 bytes
			a[i] = b.a[i];
		};
	};
	*/
	__device__ __forceinline__ void operator = (const type val){
		for (int i = 0; i < size; ++i){
			a[i] = val;
		};
	};
	__device__ __forceinline__ small_array operator + (const small_array & b)const{
		small_array c;
		for (int i = 0; i < size; ++i){
			c[i] = a[i] + b[i];
		};
		return c;
	};
	__host__ __device__ __forceinline__ small_array& operator += (const small_array & b){
		for (int i = 0; i < size; ++i){
			a[i] += b[i];
		};
		return *this;
	};
	__device__ __forceinline__ small_array operator + (const type val)const{
		small_array c;
		for (int i = 0; i < size; ++i){
			c[i] = a[i] + val;
		};
		return c;
	};
	__device__ __forceinline__ small_array& operator += (const type val){
		for (int i = 0; i < size; ++i){
			a[i] += val;
		};
		return *this;
	};
	__device__ __forceinline__ small_array operator - (const small_array & b)const{
		small_array c;
		for (int i = 0; i < size; ++i){
			c[i] = a[i] - b[i];
		};
		return c;
	};
	__device__ __forceinline__ small_array& operator -= (const small_array & b){
		for (int i = 0; i < size; ++i){
			a[i] -= b[i];
		};
		return *this;
	};
	__device__ __forceinline__ small_array operator - (const type val)const{
		small_array c;
		for (int i = 0; i < size; ++i){
			c[i] = a[i] - val;
		};
		return c;
	};
	__device__ __forceinline__ small_array& operator -= (const type val){
		for (int i = 0; i < size; ++i){
			a[i] -= val;
		};
		return *this;
	};
	__device__ __forceinline__ small_array& operator *= (const type val){
		for (int i = 0; i < size; ++i){
			a[i] *= val;
		};
		return *this;
	};
	__device__ __forceinline__ small_array operator -()const{
		small_array c;
		for (int i = 0; i < size; ++i){
			c[i] = -a[i];
		};
		return c;
	};
	__device__ __forceinline__ type min()const{
		/*
		type v = a[0];
		for (int i = 1; i < size; ++i){
			v = ::min(v, a[i]);
		};
		 */
		static_assert(size >= 2, "size must be at least 2");
		type v1 = a[0];
		type v2 = a[(size+1)/2];
		for(int i = 1; i< (size+1)/2; ++ i){
			v1 = ::min(v1, a[i]);
		};
		for(int i = (size+1)/2 +1; i<size; ++ i){
			v2 = ::min(v2, a[i]);
		};
		return ::min(v1,v2);
	};
	__device__ __forceinline__ void ldg(type * ptr, const small_array<int, size> & offset){
		for (int i = 0; i < size; ++i){
			(*this)[i] = __ldg(ptr + offset[i]);
		};
	};
	__device__ __forceinline__ void ldg(const type * ptr, const int stride){
		for (int i = 0; i < size; ++i){
			(*this)[i] = __ldg(ptr);
			ptr += stride;
		};
	};
	__device__ __forceinline__ void ld(type * ptr, const small_array<int, size> & offset){
		for (int i = 0; i < size; ++i){
			(*this)[i] = ptr[offset[i]];
		};
	};
	__device__ __forceinline__ void ld(const type * ptr, const int stride){
		for (int i = 0; i < size; ++i){
			(*this)[i] = *ptr;
			ptr += stride;
		};
	};
	__device__ __forceinline__ void st(type * ptr, const small_array<int, size> & offset)const{
		for (int i = 0; i < size; ++i){
			ptr[offset[i]] = (*this)[i];
		};
	};
	__device__ __forceinline__ void st(type * ptr, const int stride){
		for (int i = 0; i < size; ++i){
			*ptr = (*this)[i];
			ptr += stride;
		};
	};
};

template <typename type, int size>
__device__ __forceinline__
small_array<type, size> min(const small_array<type, size> & a, const small_array<type, size> & b){
	small_array<type, size> c;
	for (int i = 0; i < size; ++i){
		c[i] = ::min(a[i], b[i]);
	};
	return c;
};

template <typename type, int size>
__device__ __forceinline__
small_array<type, size> min(const small_array<type, size> & a, const type val){
	small_array<type, size> c;
	for (int i = 0; i < size; ++i){
		c[i] = ::min(a[i], val);
	};
	return c;
};
