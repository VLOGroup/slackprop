#pragma once

#include <iu/ndarray/ndarray_ref.kernel.h>
#include "slack_prop.kernel.h"
#include "slp_util.h"

#ifndef  __CUDA_ARCH__
	#include <limits>
	#define fINF (std::numeric_limits<float>::infinity())
#endif

//! common stuff to host and device code

//settings
// number of labels must factor as LABELS_PACK*LABELS_GROUP

//optimized for image size:
#define LABELS_PACK 8 // results in instruction level parallelism, min 3
#define MAX_CHAIN_LENGTH 3000
#define CHUNK 6 // unrolled processing in the last levels (min:3)

//optimized for speed:
//#define LABELS_PACK 4
//#define CHUNK 5

#define LABELS_GROUP (global.labels_group)// must be power of 2 and divide 32
#define NUM_LABELS (LABELS_PACK*LABELS_GROUP)
#define MAX_LABELS (LABELS_PACK*32)

#define WARP_SIZE 32
#define LABEL_GROUPS_PER_WARP (WARP_SIZE/LABELS_GROUP)
#define MAX_LABEL_GROUPS_PER_WARP 32
#define MAX_LINES_PER_BLOCK (32)
#define MAX_SEGMENTS (2*cpow_up(divup(MAX_CHAIN_LENGTH,CHUNK))) // total number of segments in all levels, approx = 2*(MAX_CHAIN_LENGTH/CHUNK)
#define MAX_SEG_LIST (log2up(divup(MAX_CHAIN_LENGTH,CHUNK)) + CHUNK )


template<typename type, int n> class small_array;
template<typename type> struct type_expand;


template<> struct type_expand<small_array<int, 4> >{
	typedef int s_type;
	static const int n = 4;
};

template<> struct type_expand<small_array<int, 8> >{
	typedef int s_type;
	static const int n = 8;
};

template<> struct type_expand<small_array<float, 4> >{
	typedef float s_type;
	static const int n = 4;
};

template<> struct type_expand<small_array<float, 8> >{
	typedef float s_type;
	static const int n = 8;
};


typedef small_array<float, LABELS_PACK> warp_packf;
typedef small_array<int, LABELS_PACK> warp_pack_int;
//_________________________________________________________________________________________________
struct pw_term_device{
public:
	//float L1;
	//float L2;
	float delta;
	//float slope;
	int cutoff;
public:
	__host__ __device__ pw_term_device(){};
	__host__ __device__ pw_term_device(float L1, float L2, int delta = 2){
		//this->L1 = L1;
		//this->L2 = L2;
		this->delta = delta;
		//slope = (L2 - L1) / (delta - 1);
		// if delta is < LABELS_PACK, this will just exchange to neighbors, otherwise prefix min within delta/LABELS_PACK
		cutoff = divup(delta, LABELS_PACK);
	};
	__device__ __forceinline__ warp_packf message(const warp_packf & cm, const float * W, bool dir);
	__device__ __forceinline__ warp_packf message_L1L2(const warp_packf & cm, const float * W, bool dir);
	__device__ __forceinline__ warp_packf message_TTV(const warp_packf & cm, const float * W, bool dir);
	__device__ __forceinline__ void handshake(const warp_packf & c1, warp_packf & cm1, const warp_packf & c2, warp_packf & cm2, const float coeff, const float * W, bool dir = true);
};

//_________________________________________________________________________________________________
struct global_info_cmem{
	// pointers to outputs and other junk
	kernel::ndarray_ref<int, 2> x_out;
	kernel::ndarray_ref<float, 2> LB_out;
	kernel::ndarray_ref<warp_packf, 2> C_in;
	kernel::ndarray_ref<warp_packf, 2> C_total;
	kernel::ndarray_ref<warp_packf, 2> C_out;
	kernel::ndarray_ref<warp_packf, 2> M_in;
	pw_term_device pw;
	int labels_group;
	kernel::ndarray_ref<float, 2> weights;
	//enum { reminder, total_withheld, total_free, split, recombine, resolve} cost_to_return;
	return_types cost_to_return;
public:
	int shfl_up_c;
	int shfl_dwn_c;
	__host__ __device__ global_info_cmem(){};
};

class segment_info_cmem{ // info per varied x offset
public:
	int x1; // segment placement
	int x2;
	float coeff;
	struct{
        short int pos;
    } M_list[MAX_SEG_LIST];
public:
	__host__ __device__ __forceinline__ int dir()const{
		return (x2 >= x1) ? 1 : -1;
	};
	__host__ __device__ __forceinline__ int size()const{
		return abs(x2 - x1) + 1;
	};
	__host__ __device__ bool operator < (const segment_info_cmem & b)const{
		assert((x1 < b.x1) == (x2 < b.x2));
		return (x1 < b.x1);
	};
};

//________________access to constant memory__________________________
global_info_cmem * get_global_dev_ptr();
segment_info_cmem * get_segments_dev_ptr();

////_________________________kernels___________________________________
//__global__ void work_pack(int seg_level_begin, bool handshake);
//__global__ void unrolled_sm(int seg_level_begin);
//__global__ void handshake_k(int x, float coeff);

void launch_work_pack(const dim3 & dimGrid, const dim3 & dimBlock, int seg_level_begin, bool handshake);
void launch_unrolled_sm(const dim3 & dimGrid, const dim3 & dimBlock, int seg_level_begin);
void launch_handshake_k(const dim3 & dimGrid, const dim3 & dimBlock, int x, float coeff);
