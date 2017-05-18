// This file is part of slack-prop.
//
// Copyright (C) 2017 Sasha Shekhovtsov <shekhovtsov at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// slack-prop is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// slack-prop is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "slp_kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iu/ndarray/error_cuda.h>

//__constant__ __device__ int M_lookup[MAX_CHAIN_LENGTH];
//const __constant__ __device__ global_info_cmem global;

/*
int * get_M_lookup_dev_ptr(){
	int * dev_ptr;
	//cudaGetSymbolAddress((void**)&dev_ptr, M_lookup);
	cuda_check_error();
	return dev_ptr;
};
*/

global_info_cmem * get_global_dev_ptr(){
	global_info_cmem * dev_ptr;
	cudaGetSymbolAddress((void**)&dev_ptr, global);
	cuda_check_error();
	return dev_ptr;
};

//_______________________________________________________________
__constant__ __device__ segment_info_cmem_d cmem_segments[MAX_SEGMENTS];

segment_info_cmem * get_segments_dev_ptr(){
	segment_info_cmem_d * dev_ptr;
	cudaGetSymbolAddress((void**)&dev_ptr, cmem_segments);
	cuda_check_error();
	return dev_ptr;
};

//_________________________________________________________________
__device__ void print_pack(const char * name, const warp_packf & a, int debug_lvl){
	if (DEBUG_LVL >= debug_lvl){
		if (LABELS_PACK == 4){
			printf("%s[b%i][t%i] = [%3.2f %3.2f %3.2f %3.2f]\n", name, blockDim.z*blockIdx.z + threadIdx.z, threadIdx.x, a[0], a[1], a[2], a[3]);
		};
		if (LABELS_PACK == 3){
			printf("%s[b%i][t%i] = [%3.2f %3.2f %3.2f]\n", name, blockDim.z*blockIdx.z + threadIdx.z, threadIdx.x, a[0], a[1], a[2]);
		};
		if (LABELS_PACK == 2){
			printf("%s[b%i][t%i] = [%3.2f %3.2f]\n", name, blockDim.z*blockIdx.z + threadIdx.z, threadIdx.x, a[0], a[1]);
		};
	};
};

__device__ void print_pack(const char * name, const float & a, int debug_lvl){
	if (DEBUG_LVL >= debug_lvl){
		printf("%s[b%i][t%i] = {%3.2f}\n", name, blockDim.z*blockIdx.z + threadIdx.z, threadIdx.x, a);
	};
};


__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
work_pack(int seg_level_begin, bool handshake){
	//
	const int line_number = blockIdx.y*blockDim.y + threadIdx.y; // global line number
	if (line_number >= global.C_in.sz[1]){// cut out of bounds lines
		return;
	};
	const int tbl_idx = seg_level_begin + blockIdx.z*blockDim.z + threadIdx.z;
	//
	const segment_info_cmem_d & sg = cmem_segments[tbl_idx];
    const short int * __restrict__ pp = &sg.M_list[1].pos;
	//
	pw_term_device pw = global.pw;
	const warp_packf * __restrict__ pC = &sg.C_in(0, line_number); // + threadIdx.x;
	//dslice<warp_packf,1> pC = sg.C_in_slice(0, line_number); // + threadIdx.x;
	warp_packf * __restrict__ pM = &sg.M_in(0, line_number); // + threadIdx.x;
	/*
	__shared__ const int C_stride0b = 0;
	__shared__ const int M_stride0b = 0;
	__shared__ const int seg_size = 0;
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
	};
	__syncthreads();
	*/

	const auto C_stride0 = sg.C_stride0();
	const auto M_stride0 = C_stride0;
	const int seg_size = sg.size();

	float * __restrict__ W = sg.weights(0, line_number);
	if (threadIdx.x == 0 && DEBUG_LVL >= 2){
		//printf("Big interval[b%i]: [%i -> %i] \n", threadIdx.z + blockIdx.z*blockDim.z, cmem_segments[tbl_idx].x1, cmem_segments[tbl_idx].x2);
	};
	warp_packf c1; // costs pack
	warp_packf cm1;
	if (handshake){
		cm1 = *pM;
	} else{
		cm1 = *pC; // initialization
		*pM = cm1;
	};
	pC += C_stride0;
	pM += M_stride0;
	float * __restrict__ W_ahead = W;
	W_ahead += sg.weights_stride0();
	// a pass
	//TODO: remove reading ahead
	int x1 = sg.x1 + sg.dir();
	int wpos = *pp;
	warp_packf c1_ahead = *pC;
	for (int x = seg_size - 1; x > 0; --x){ // count down seg_size
		//c1 = *pC;
		c1 = c1_ahead;
		pC += C_stride0;
		c1_ahead = *pC;
		warp_packf m1 = pw.message(cm1, W, sg.dir());
		// normalize
		 //float  full_min = group_min(m1.min());
		 //m1 -= full_min;
		//
		if (DEBUG_LVL >= 2){
			warp_packf M = m1 + *pM; // full min-marginal
			float minF = group_min(M.min(),LABELS_GROUP);
			if (threadIdx.x == 0){
				//printf("minM[b%i]=%f\n", blockDim.z*blockIdx.z + threadIdx.z, minF);
			};
		};
		cm1 = m1 + c1; // (left) min-marginal + cost
		// normalize
		//float  full_min = group_min(cm1.min());
		//cm1 -= full_min;
		//
		if(1){
			if (x1 == wpos){
				// write transaction: save m1+cost here
				*pM = cm1;
				++pp;
				wpos = *pp;
			};
		}else{
			*pM = cm1;
		};
		pM += M_stride0;
		W = W_ahead;
		W_ahead += sg.weights_stride0();
		x1 += sg.dir();
	};
	// gather c1, c2, cm1, cm2 in a single warp for handshake, assume the meeting message is in global mem
	if (handshake){
		const float coeff = sg.coeff;
		warp_packf c2 = *pC;
		warp_packf cm2 = *pM;
		// handshake
		pw.handshake(c1, cm1, c2, cm2, coeff, W, sg.dir() );
		// store
		*pM = cm2;
		pM -= M_stride0;
		*pM = cm1;
	};
	if (threadIdx.x == 0 && tbl_idx == seg_level_begin){
		//print_array(pf_stream(2),"MM=", global.M_in.recast<float>(), 2);
	};
};


// blockDim (LABELS_GROUP, LABEL_GROUPS_PER_WARP x A, 1)
// blockIdx.x - superlabel
// blockIdx.y - line number within block
// blockIdx.z - 1
// gridDim (1, ny, 1)
__global__ void handshake_k(int x, float coeff){
	pw_term_device pw = global.pw;
	const int line_number = blockIdx.y*blockDim.y + threadIdx.y; // global line number
	if (line_number >= global.C_in.sz[1]){// cut out of bounds lines
		return;
	};
	warp_packf * __restrict__ C1 = global.C_in.ptr(x, line_number) + threadIdx.x;
	warp_packf * __restrict__ C2 = global.C_in.ptr(x+ 1, line_number) + threadIdx.x;
	warp_packf * __restrict__ M1 = global.M_in.ptr(x, line_number) + threadIdx.x;
	warp_packf * __restrict__ M2 = global.M_in.ptr(x + 1, line_number) + threadIdx.x;
	//int C_stride0b = global.C_in.stride_bytes(0);
	float * __restrict__ W = global.weights.ptr(x, line_number); // shared across groug / threadIdx.x
	//
	//warp_packf c1 = *C;
	//warp_packf c2 = *(warp_packf*)((char*)C + C_stride0b);
	//warp_packf cm1 = *M;
	//warp_packf cm2 = *(warp_packf*)((char*)M + C_stride0b);
	//if(threadIdx.x==0)printf("handshake w:%f\n",weight);
	warp_packf cm1 = *M1;
	warp_packf cm2 = *M2;
	pw.handshake(*C1, cm1, *C2, cm2, coeff, W, 1);
	*M1 = cm1;
	*M2 = cm2;
	if (threadIdx.x == 0){
		//print_array("MM=", global.M_in.recast<float>(), NUM_LABELS, 2);
	};
};

void launch_work_pack(const dim3 & dimGrid, const dim3 & dimBlock, int seg_level_begin, bool handshake){
	work_pack <<< dimGrid, dimBlock >>>(seg_level_begin, handshake);
};

void launch_handshake_k(const dim3 & dimGrid, const dim3 & dimBlock, int x, float coeff){
	handshake_k <<< dimGrid, dimBlock >>>(x, coeff);
};


//______________________________________unrolled levels in SM___________________________________________
//__shared__ __device__ warp_packf SC[CHUNK][WARP_SIZE * BLOCK_WARPS]; // costs
//__shared__ __device__ int Sx[MAX_LINES_PER_BLOCK]; // some kind of temp
//__shared__ __device__ float weights[CHUNK][MAX_LINES_PER_BLOCK]; // chunk weights

//struct t_shared{

__shared__  __device__ int shared_Sx[MAX_LINES_PER_BLOCK]; // some kind of temp
__shared__  __device__ float * __restrict__ shared_weights[CHUNK][MAX_LINES_PER_BLOCK]; // pointers to chunk weights
__shared__  __device__ dslice<warp_packf, 1> shared_C_total[MAX_LINES_PER_BLOCK];
__shared__  __device__ dslice<warp_packf, 1> shared_C_out[MAX_LINES_PER_BLOCK];
__shared__  __device__ dslice<int, 1> shared_x_out[MAX_LINES_PER_BLOCK];
__shared__  __device__ dslice<float, 1> shared_LB_out[MAX_LINES_PER_BLOCK];
//	__host__ __device__ t_shared(){};
	//};
//__shared__  __device__ t_shared shared;

struct chain_worker_SM_ctx{
public:
	int segment_ifo_idx;
	__device__ __forceinline__  const segment_info_cmem_d & cmem(){
		return cmem_segments[segment_ifo_idx];
	};
	pw_term_device pw;
	// vars that are needed for computation
	//__shared__ __device__ warp_packf SM[CHUNK][WARP_SIZE * BLOCK_WARPS]; // messages
	warp_packf SM[CHUNK]; // message pack per thread
	warp_packf SC[CHUNK]; // message pack per thread
	//warp_packf CT[CHUNK];
	//int pack_id;  // where pack for this thread is localed in shared mem - per block (combined threadIdx.x, threadIdx.y and threadIdx.z)
	int line_number;  // line number for the whole level, differs per threadIdx.y, threadIdx.z, blockIdx.y
	//int warp_id;  // to index into Sx;
	int local_line_number;
public:
	//_____________________________________________________
	__device__ __forceinline__ warp_packf & get_M(int x){
		return SM[x];
	}
	__device__ __forceinline__ warp_packf & get_C_in(int x){
		return SC[x];
	}
	__device__ __forceinline__ warp_packf & get_C_out(int x){
		return shared_C_out[local_line_number].ptr(x)[threadIdx.x];
	}
	__device__ __forceinline__ warp_packf & get_C_total(int x){
		return shared_C_total[local_line_number].ptr(x)[threadIdx.x];
	}

	//_____________________________________________________
	void load_data(int x, int y_block = 0){
		//  const int line_number = blockIdx.y*gridDim.y +  *blockDim.y + threadIdx.y;
		//SC[x] = &cmem().C_in(x, line_number);
		//SC[x] = &cmem().C_in(x, line_number);
	};

	template <int x>
	__device__ RINLINE void finalize(warp_packf F){
		static_assert(x >= 0 && x < CHUNK, "range check");
		// mm is full decorrelated min-marginal
		// find min and LB
		if (blockIdx.z == 0 && threadIdx.z == 0 && DEBUG_LVL >= 3){
			//pf_stream(1) << "x=" << x << ":";
			//print_array("F=", &F[0], LABELS_PACK, 1);
		};
		float minF = group_min(F.min(),LABELS_GROUP);
		//printf("minF[t%i]=%f\n",threadIdx.x,minF);
		//find where min is achieved
		//__syncthreads();
		/*
		for (int i = 0; i < LABELS_PACK; ++i){
			if (F[i] == minF){ // race condition, any from the group fits
				shared_Sx[local_line_number] = (threadIdx.x * LABELS_PACK + i); // label value
			};
		};
		*/
		int min_i = 1e5;
		for (int i = 0; i < LABELS_PACK; ++i) {
			//if (F[i] == minF){
			if (F[i] <= minF + def_abs(minF)*1e-6f){
				min_i = (threadIdx.x * LABELS_PACK + i); // label value
			};
		};
		min_i = group_min(min_i,LABELS_GROUP);

		//__syncthreads();
		if (threadIdx.x == 0){ //per pixel (combined threadIdx.y and threadIdx.z)
			// calculate where to store this crap
			//cmem().x_out(x, line_number) = shared_Sx[local_line_number];
			//cmem().LB_out(x, line_number) = minF;
			//*(shared_x_out[local_line_number].ptr(x)) = shared_Sx[local_line_number];
			*(shared_x_out[local_line_number].ptr(x)) = min_i;
            *(shared_LB_out[local_line_number].ptr(x)) = minF;
		};
		/*
		// can sum minF across entire block, first across groups
		#pragma unroll
		for (int i = LABELS_GROUP; i < WARP_SIZE; i = i << 1){
		minF += __shfl_xor(minF, i, WARP_SIZE);
		};
		*/
		// new output costs
		//warp_packf c1 = SC[x][pack_id];
		warp_packf c1 = get_C_in(x);
		//F -= c1;
		//F = F - c1 - minF;
		//from c1 can take out F, remains C_total - c1 + F;
		//warp_packf c_total = (shared_C_total[local_line_number].ptr(x))[threadIdx.x];
		warp_packf c_total = get_C_total(x);
		//warp_packf c_total = cmem().C_total(x, line_number);
		/*
		warp_packf c_total_r = cmem().C_total(x, line_number);
		for (int i = 0; i < LABELS_PACK; ++i){
		if (c_total[i] != c_total_r[i]){
		asm("trap;");
		};
		};
		*/
		//cmem().C_out(x, line_number) = cmem().C_total(x, line_number) - c1 + F;
		//cmem().C_out(x, line_number) = c_total - c1 + F;

		/*
		warp_packf H2 = get_C_out(x); // saved H2
		warp_packf H1 = c1 - F; // hold out in this subproblem = H1
		get_C_out(x) = H1;
		void * ptr = (void *)shared_C_total[local_line_number].ptr();
		if( size_t(ptr) %16 != 0 && local_line_number == 0){
			void *beg = (void*)global.C_total.ptr(0);
			int diff = (char*)ptr - (char*)beg;
			printf("block %i, thread %i, deteced missaligned: x=%i, dir=%i, beg = %p, ptr = %p, diff=%i\n", blockIdx.z, threadIdx.x, x, cmem().dir(), beg, ptr, diff);
		}

		get_C_total(x) = F + H2; // cost for the second subproblem
		return;
		 */

		switch (global.cost_to_return){
		case return_types::reminder:
		{
			get_C_out(x) = (c_total -= c1) += F; // saved hold out in the other problem + new free
			break;
		}
		case return_types::total_free:
		{
			get_C_out(x) = F;
			break;
		}
		case return_types::total_withheld:
		{
			get_C_out(x) = c_total -= F;
			break;
		}
		case return_types::resolve:
		{
			//for(int i=0;i<LABELS_PACK;++i)if( fabs(c1[i] - c_total[i]) > 1e-5 )return;
			warp_packf H2 = get_C_out(x); // saved H2
			warp_packf H1 = c1 - F; // hold out in this subproblem = H1
			get_C_out(x) = H1;
			get_C_total(x) = F + H2; // cost for the second subproblem
			break;
		}
		case return_types::split:
		{
			warp_packf H_other = get_C_out(x); // saved H_other
			warp_packf H_my = c1 - F; // hold out in this subproblem
			//get_C_out(x) = H_other; // save other
			get_C_out(x) = H_my; // save other
			warp_packf TH = H_my + H_other;
			get_C_total(x) = -TH; // total negative hold out
			break;
		}
		case return_types::recombine:
		{
			warp_packf c2 = get_C_out(x); // saved hold out in the other problem
			get_C_total(x) = c2 + c1; // recombined total cost
			get_C_out(x) = c2 + F; // saved hold out in the other problem + new free
			break;
		}
		default:
			get_C_out(x) = FINF;
		};
		//
		//
        //load_data(x, 1);
	}
	//______________________________________
	template <int dir>
	__device__ __forceinline__ float * sm_weight(const int x){
		if (dir > 0){
			return shared_weights[x][local_line_number];
		} else{
			return shared_weights[x - 1][local_line_number];
		};
	}

	template <int x1, int x2>
	__device__ warp_packf RINLINE chain_cm(){
		const int dir = x2 >= x1 ? 1 : -1;
		warp_packf cm1 = SM[x1];
		// a pass
//#pragma unroll
		for (int x = x1; x != x2;){
			float * __restrict__ W = sm_weight<dir>(x);
			/*
			if (dir > 0){
				weight = weights[x][lolac_line_number];
			} else{
				weight = weights[x-1][local_line_number];
			};
			*/
			warp_packf m1 = pw.message(cm1, W, dir);
			//print_pack("m1",m1,1);
			//warp_packf c1 = SC[x][pack_id];
			x += dir;
			warp_packf c1 = get_C_in(x);
			cm1 = m1 + c1;
			get_M(x) = cm1; // store cost msg
		};
		return cm1;
	}

	//_____________________________________
	template<int sz>
	__device__ RINLINE void chunk_expand(){
		// load data
		const auto C_stride0 = cmem().C_stride0();
		const auto w_stride0 = cmem().weights_stride0();
		//int C_stride1b = cmem().C_stride1_bytes();
		const warp_packf * __restrict__ C_in = &cmem().C_in(0, line_number);// + threadIdx.x; // threadIdx.x - offset in labels dim in packs
		warp_packf * __restrict__ M_in = &cmem().M_in(0, line_number);// + threadIdx.x;
		float * __restrict__ p_weight = cmem().weights(0, line_number);
		//
		if (threadIdx.x == 0){ // per line
			//shared_C_out[local_line_number]   = dslice<warp_packf, 1>(global.C_out,   cmem().x1, line_number, cmem().dir());
			shared_C_out[local_line_number]   = global.C_out.subdim<1>(line_number).offset(cmem().x1).direction(cmem().dir());
			//shared_C_total[local_line_number] = dslice<warp_packf, 1>(global.C_total, cmem().x1, line_number, cmem().dir());
			shared_C_total[local_line_number]   = global.C_total.subdim<1>(line_number).offset(cmem().x1).direction(cmem().dir());
			//shared_x_out[local_line_number] = dslice<int, 1>(global.x_out, cmem().x1, line_number, cmem().dir());
			shared_x_out[local_line_number]   = global.x_out.subdim<1>(line_number).offset(cmem().x1).direction(cmem().dir());
			//shared_LB_out[local_line_number] = dslice<float, 1>(global.LB_out, cmem().x1, line_number, cmem().dir());
			shared_LB_out[local_line_number]   = global.LB_out.subdim<1>(line_number).offset(cmem().x1).direction(cmem().dir());
		};
		static_assert(sz<=CHUNK,"bad");
		if(local_line_number>=MAX_LINES_PER_BLOCK){
			asm("trap;");
		};
		//#pragma unroll
		for (int x = 0; x < sz; ++x){
			//SC[x][ctx.pack_id] = *C_in;
			SC[x] = *(C_in + x*C_stride0);
			SM[x] = *(M_in + x*C_stride0);
			//C_in += C_stride0;
			//M_in += C_stride0;
			//CT[x] = shared_C_total[local_line_number].ptr(x)[threadIdx.x];
			if (threadIdx.x == 0){ // per pixel
				shared_weights[x][local_line_number] = p_weight;
				p_weight += w_stride0;
			};
			//print_pack("C[x]",SC[x],1);
			//print_pack("M[x]",SM[x],1);
			//if(threadIdx.x==0){
			//	printf("x[b%i]=%i : ",threadIdx.z,x);
			//	print_pack("M",SM[x],1);
			//};
		};
		/*
		__threadfence_block();
		for (int x = 0; x < sz; ++x){
			//SC[x][ctx.pack_id] = *C_in;
			warp_packf m = get_M(x);
			warp_packf c = get_C_in(x);
			warp_packf ct = get_C_total(x);
			get_C_out(x) = ct + c + m - group_min(m.min());
		};
		return;
		*/
		assert(sz >= (CHUNK + 1) / 2 && sz <= CHUNK);
		// prevent unnecessary instantiations
		const int sz1 = sz <= CHUNK ? sz : 2;
		const int sz2 = sz1 >= (CHUNK + 1) / 2 ? sz1 : 2;
		chunk_unroll<0, sz2 - 1>();
	}

	template <int x1, int x2>
	__device__ __forceinline__ void chunk_unroll(){
		if (blockIdx.x == 0 && threadIdx.z == 0 && threadIdx.y == 0 && blockIdx.y == 0 && blockIdx.z == 0){
			//pf_stream(3) << "interval [" << x1 << "->" << x2 << "]\n";
		};
		static_assert(x1 >= 0, "bad");
		static_assert(x2 >= 0, "bad");
		static_assert(x1 < CHUNK, "bad");
		static_assert(x2 < CHUNK, "bad");
		const int dir = x2 >= x1 ? 1 : -1;
		const int sz = def_abs(x2 - x1) + 1;
		static_assert(sz >= 2, "chunk size is 1 or 0");
		const int x1a = (dir > 0) ? ((x1 + x2) / 2) : ((x1 + x2 + 1) / 2); //closer to x1
		static_assert(x1a != x2, "bad");
		const int x2a = (x1a == x2) ? (x2) : (x1a + dir);
		static_assert(x1a >= 0, "bad");
		static_assert(x2a >= 0, "bad");
		//static_assert(x2a <= def_max(x1,x2), "bad");
		const int h1 = def_abs(x1a - x1) + 1;
		const int h2 = def_abs(x2a - x2) + 1;
		//static_assert(h1 + h2 == sz, "halfs do not match");
		warp_packf cm1 = chain_cm<x1, x1a>(); // msg pass from x1 to x1a
		//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("cm1=", &cm1[0], LABELS_PACK, 1);
		// handshake with x2a
		static_assert(x1a != x2a, "bah!");
		warp_packf cm2 = SM[x2a];
		//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("cm2=", &cm2[0], LABELS_PACK, 1);
		float * __restrict__ W = sm_weight<dir>(x1a);
		warp_packf m = pw.message(cm1, W, dir);  // message x1a->x2a
		//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("m=", &m[0], LABELS_PACK, 1);
		// expanded handshake
		warp_packf M = m + cm2; // full min-marginal at x2a
		//if(blockIdx.z == 0 && DEBUG_LVL>=2)print_array("M=",&M[0],LABELS_PACK,1);
		M *= (float(h1) / float(sz)); // portion to share to h1
		// share to cm1
		m = pw.message(M - m, W, -dir);
		//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("m_share=", &m[0], LABELS_PACK, 1);
		if (h1 == 1){ // left segment is of size 1
			finalize<x1>(cm1 + m);
		} else{ // compute cm cost for left segment
			//warp_packf c1 = SC[x1a][pack_id];
			warp_packf c1 = get_C_in(x1a);
			cm1 = c1 + m;
			//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("cm1_st=", &cm1[0], LABELS_PACK, 1);
			SM[x1a] = cm1;  //store
		};
		// bounce back
		if (h2 == 1){ // right segment is of size 1
			finalize<x2>(cm2 + pw.message(-m, W, dir));
		} else{ // compute cm cost for right
			//warp_packf c2 = SC[x2a][pack_id];
			warp_packf c2 = get_C_in(x2a);
			cm2 = c2 + pw.message(-m, W, dir);
			//if (blockIdx.z == 0 && DEBUG_LVL >= 3)print_array("cm2_st=", &cm2[0], LABELS_PACK, 1);
			SM[x2a] = cm2; //store
		};
		// recursion goes outside, everything that needed to be remembered is in registers / shared mem
		chunk_unroll < (h1 > 1) ? x1a : 0, (h1 > 1) ? x1 : 0 > (); // backwards
		chunk_unroll < (h2 > 1) ? x2a : 0, (h2 > 1) ? x2 : 0 > (); // forward
	}
	//
	__device__ __forceinline__ void chunk_unroll_dispatch(int size){
		// need to serve cases in [(CHUNK+1)/2 ... CHUNK]
		assert(size >= (CHUNK + 1) / 2 && size <= CHUNK);
		static_assert(CHUNK >= 3, "minimum size to expand");
		switch (size){
			//case 1: chunk_expand<1>(); break;
		case 2: chunk_expand<2>(); break;
		case 3: chunk_expand<3>(); break;
#if CHUNK>3
		case 4: chunk_expand<4>(); break;
#endif
#if CHUNK>4
		case 5: chunk_expand<5>(); break;
#endif
#if CHUNK>5
		case 6: chunk_expand<6>(); break;
#endif
#if CHUNK>6
		case 7: chunk_expand<7>(); break;
#endif
#if CHUNK>7
		case 8: chunk_expand<8>(); break;
#endif
#if CHUNK>8
#error no chunk expansion
#endif
			//case 9: chunk_expand<9>(); break;
			//case 10: chunk_expand<10>(); break;
			//case 11: chunk_expand<11>(); break;
			//case 12: chunk_expand<12>(); break;
			//case 13: chunk_expand<13>(); break;
			//case 14: chunk_expand<14>(); break;
			//case 15: chunk_expand<15>(); break;
			//case 16: chunk_expand<16>(); break;
        default: //pf_stream(1) << "segment not dispatched\n"; asm("trap;");
        	assert(false);
		};
	};
};

template<>
__device__ __forceinline__ void chain_worker_SM_ctx::chunk_unroll<0, 0>(){// terminates recusrion
};

// launch config
// blockDim (LABELS_GROUP, LABEL_GROUPS_PER_WARP x A, B) s.t. A*B = BLOCK_WARPS
// blockIdx.x - superlabel
// blockIdx.y - line number within block
// blockIdx.z - offset
// gridDim (1, ny, nz)

__global__ void __launch_bounds__(256, 2)
unrolled_sm(int seg_level_begin){
	//TODO: loop to preload date for the next block (iterate over blocks instead of parallel - now there's to many)
	//TODO:
	// init context
	chain_worker_SM_ctx ctx;
	ctx.pw = global.pw;
	const int line_number = blockIdx.y*blockDim.y + threadIdx.y; // global line number
	if (line_number >= global.C_in.sz[1]){// cut out of bounds lines
		return;
	};
	const int tbl_idx = seg_level_begin + blockIdx.z*blockDim.z + threadIdx.z;
	//if(tbl_idx == seg_level_begin && threadIdx.x == 0){
	//	print_array("MM=",global.M_in.recast<float>(),NUM_LABELS,1);
	//};
	ctx.segment_ifo_idx = tbl_idx;
	ctx.line_number = line_number; // global line number
	//ctx.warp_id = (threadIdx.z*blockDim.y + threadIdx.y) / WARP_SIZE;
	ctx.local_line_number = threadIdx.z*blockDim.y + threadIdx.y;
	//ctx.pack_id = (threadIdx.z * blockDim.y + threadIdx.y) * LABELS_GROUP + threadIdx.x; // pack index within block, since shared mem is per block
	const int sz = ctx.cmem().size();
	//work in shared mem
	ctx.chunk_unroll_dispatch(sz);
}

void launch_unrolled_sm(const dim3 & dimGrid, const dim3 & dimBlock, int seg_level_begin){
	unrolled_sm <<< dimGrid, dimBlock >>>(seg_level_begin);
}
