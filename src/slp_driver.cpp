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

#include <string>
#include <iostream>
#include <sstream>

#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <assert.h>
#include <string>

#ifndef DEBUG_LVL
#define DEBUG_LVL -1
#endif

#include <iu/ndarray/ndarray.h>
#include <iu/ndarray/ndarray_io.h>
#include <iu/ndarray/ndarray_print.h>

#include "slack_prop.h"
//
#include "slp_util.h"
#include "slp_kernel.h"


/*
void cmem_configure(ndarray_ref<float, 2> C1, ndarray_ref<float, 2> C2, ndarray_ref<float, 2> C_total, ndarray_ref<int, 2> sol, ndarray_ref<int, 2> LB){
	// for each block to launch prepare its config in shared mem
};
 */

int hob(int num){
	if (!num)return 0;
	int ret = 1;
	while (num >>= 1){
		ret <<= 1;
	};
	return ret;
};

int log2i(int num){
	int ret = 0;
	while (num >>= 1){
		++ret;
	};
	return ret;
};


struct work_plan_level{
	std::vector<segment_info_cmem> seg_info;
	int start;
};

//________________________________________________________________________________
class slack_prop_driver{
public: // input
	ndarray_ref<float, 3> C_total; // input total cost
	ndarray_ref<float, 3> C1;      // input current reparametrized cost (withheld + free)
	ndarray_ref<float, 3> C2;      // output cost (reminder = total - withheld), may point to the same buffer as C2 - inplace calculation
	ndarray<float, 3> M;	       // chain internal messages (temp)
	ndarray_ref<int, 2> sol;
	ndarray_ref<float, 3> weights;
	pw_term_device pw;
public: // useful
	int n_levels;
	int leaf_level;
	int n_chains;
	int N;
	int K;
	double LLB;
	slack_prop_ops ops;
private:
	std::vector<work_plan_level> plan;
	std::vector<segment_info_cmem> all_segments;
	ndarray<float, 2> _LB;
	//ndarray_ref<float, 2> LB_s;
	std::vector<int> M_lookup;
	//std::vector<char> M_dir;
	//int M_mask[32][MAX_CHAIN_LENGTH];
	//enum {w_left, w_right, r, rw_left, rw_right} m_type;
	//std::string M_mask;
	std::vector<char> M_mask;
public: // more constants
	global_info_cmem global;
public:
	slack_prop_driver(){};
	void init(ndarray_ref<float, 3> C1, ndarray_ref<float, 3> C2, ndarray_ref<float, 3> C_total, ndarray_ref<int, 2> sol, pw_term_device pw, ndarray_ref<float,3> weights){
		//std::cout << C1.size() << " strides: " << C1.stride_bytes() << "\n";
		//check_all(C1);
		if(DEBUG_LVL>=1){
			std::cout << "C1" << C1 << "\n";
			std::cout << "C2" << C2 << "\n";
			std::cout << "C_total" << C_total << "\n";
			std::cout << "sol" << sol << "\n";
			std::cout << "weights" << weights << "\n";
		};
		this->C1 = C1;
		this->C2 = C2;
		this->C_total = C_total;
		this->sol = sol;
		this->pw = pw;
		//std::cout << "C1=" << C1 << "\n";
		this->K = C1.size(0);
		//pf_stream(0)<<"K="<<K<<"\n";
		//pf_stream(0)<<"C1.size = " << C1.size() << "\n";
		runtime_check(C1.size() == C2.size());
		runtime_check(C2.size() == C_total.size());
		//std::cout << "weights = "<< weights << "\n";
		this->weights = weights;
		init();
	};
	void set_C_total(ndarray_ref<float, 3> C_total){
		this->C_total = C_total;
	};
	void set_C_in(ndarray_ref<float, 3> C_in){
		this->C1 = C_in;
	};
private:
	void init(){
		cuda_check_error();
		// check assumptions on K
		if (K < LABELS_PACK) throw_error() << "K(number of lables) must be at least LABELS_PACK="<< LABELS_PACK << "\n";
		if (K > MAX_LABELS)throw_error() << "K(number of lables) cannot be more than MAX_LABELS=" << MAX_LABELS << "\n";
		for (int v = K / LABELS_PACK; v > 1; v = v/2 ){
			if (v % 2 == 1){
				throw_error() << "K /LABELS_PACK = " << K / LABELS_PACK << "  must be power of two";
			};
		};

		//K = NUM_LABELS;
		N = C1.size(1); // chain length
		if (N < CHUNK)throw_error() << "N(chain length) cannot be less than CHUNK (=" << CHUNK << ") (plan issue)\n";
		n_chains = C1.size(2);
		pf_stream(1) << "Length:" << N << " n_chains:" << n_chains << " \n";
		pf_stream(1) << "Init mem\n";
		init_mem();
		global.cost_to_return = return_types::reminder;
		init_global();
		pf_stream(0) << "Init plan\n";
		n_levels = log2i(N) + 1;
		// total segments, at n_levels - 3
		leaf_level = n_levels - 1; // just an estimate -- push all leafs there
		pf_stream(0) << "Total levels: " << n_levels << " leaf_level=" << leaf_level << "\n";
		//int total_segments = 2 + (N * 2) / CHUNK;
		plan.resize(leaf_level + 1);
		// generate levels
		int x1 = 0;
		int x2 = N - 1;
		const int x1a = (x1 + x2) / 2;
		const int x2a = x1a + 1;
		// plan 0
		segment_info_cmem sg;
		sg.x1 = x1;
		sg.x2 = x1a;
		sg.coeff = float(x1a - x1 + 1) / float(N);
		//sg.sz = sz;
		plan[0].seg_info.push_back(sg);
		sg.x1 = x2;
		sg.x2 = x2a;
		sg.coeff = 0;
		plan[0].seg_info.push_back(sg);
		// plan 1 and down on
		init_segment(1, x1a, x1);
		init_segment(1, x2a, x2);
		// sort segments in plan
		//std::cout << sg.x1 << " " << sg.x2 <<"\n";
		int total_segments = 0;
		for (int l = 0; l < plan.size(); ++l){
			total_segments += (int)plan[l].seg_info.size();
			std::sort(plan[l].seg_info.begin(),plan[l].seg_info.end());
		};

		pf_stream(0) << "total segments: " << total_segments << "\n";
		if (total_segments > MAX_SEGMENTS){
			throw_error() << "Maximum number of segments is exceeded (too big image for this version or more const mem is needed)" << " total segments=" << total_segments << " < MAX_SEGMENTS=" << MAX_SEGMENTS <<"\n";
		};
		// compute mask from the plan
		compute_mask();
		// count starting index for each level, and copy to linear array
		int start = 0;
		all_segments.reserve(total_segments);
		for (int l = 0; l < plan.size(); ++l){
			int n = (int)plan[l].seg_info.size();
			pf_stream(1) << "lvl " << l << ", \t segments=" << n;
			int maxseg = 0;
			if (n>0){
				for (int i = 0; i< n; ++i){
					segment_info_cmem sg	= plan[l].seg_info[i];
					pf_stream(1) << "[" << sg.x1 << "->" << sg.x2 << "]";
					maxseg = std::max(maxseg, (int)sg.size());
					all_segments.push_back(sg);
				};
				pf_stream(1) << " max_segment_size: " << maxseg;
				//cudaMemcpyToSymbol(cmem_segments, &*plan[l].seg_info.begin(), n*sizeof(segment_info_cmem), start*sizeof(segment_info_cmem));
				//cudaMemcpy(get_segments_dev_ptr() + start, &plan[l].seg_info[0], n*sizeof(segment_info_cmem), cudaMemcpyHostToDevice);
				//cuda_check_error();
			};
			pf_stream(1) << "\n";
			plan[l].start = start;
			start += n;
		};
	};

	void load_const(){
		pf_stream(1) << "Load Const\n";
		cudaMemcpy(get_global_dev_ptr(), &global, sizeof(global), cudaMemcpyHostToDevice);
		cuda_check_error();
		cudaMemcpy(get_segments_dev_ptr(), &all_segments[0], (int)all_segments.size()*sizeof(segment_info_cmem), cudaMemcpyHostToDevice);
		cuda_check_error();
	};
public:
	void execute(){
		cuda_check_error();
		init_global();
		load_const();
		for (int l = 0; l < plan.size(); ++l){
			work_level(l);
			if(l == ops.debug_terminate_level){
				break; //exit after level l
			};
		};
		cudaDeviceSynchronize();
		cuda_check_error();
		//std::cout << "_LB" << _LB << "\n";
		//		print_array("LB=", ndarray_ref<float, 2>(LB._host, N, 1, n_chains, N), 1);
//#if DEBUG_LVL >= 1
		if(ops.compute_LB){
			ndarray<float,2>_LB;
			_LB.create<memory::CPU>(this->_LB);
			copy_data(_LB, this->_LB);
			pf_stream(1) << "LB=";
			LLB = 0;
			for (int i = 0; i < _LB.size(0); ++i){
				for (int j = 0; j < _LB.size(1); ++j){
					LLB += _LB(i, j);
				};
			};
			pf_stream(1) << LLB << "\n";
		}else{
			LLB = 0;
		};
//#endif
	};

	~slack_prop_driver(){
	};
private:
	void init_mem(){
		//allocate space for messages
		M.create<memory::GPU>(C1);
		_LB.create<memory::GPU>(C1.size().erase<0>());
		// space for LB
		//LB.init(N*n_chains);
		//LB_s = ndarray_ref<float, 2>(LB._device, N, 1, n_chains, N);
		//LB = sol.recast<float>(); // copy size and strides
		//cudaMalloc((void**)&LB._beg, N * n_chains * sizeof(float));
	};

	void init_segment(int lvl, int x1, int x2){
		//pf_stream(1) << "["<<x1 << "->" << x2 << "]\n";
		// add job to level lvl, or to the leaf if not subdivided
		const int dir = x2 > x1 ? 1 : -1;
		const int sz = abs(x2 - x1) + 1;
		segment_info_cmem sg;

		// mark used message entries
		//M_lookup[x1] = 1;
		//M_lookup[x2] = 1;
		//for (int x = x1;; x += dir){
		//	M_lookup[x] = 
		//	if (x == x2)break;
		//};
		// subdivide
		//assert(sz >= 2);
		if (sz > CHUNK){ // still too big for a leaf, subdivide
			const int x1a = (dir>0) ? ((x1 + x2) / 2) : ((x1 + x2 + 1) / 2); //closer to x1
			assert(x1a != x2);
			const int x2a = (x1a == x2) ? (x2) : (x1a + dir);
			assert(x2a <= std::max(x1, x2));
			const int h1 = abs(x1a - x1) + 1;
			const int h2 = abs(x2a - x2) + 1;
			assert(h1 + h2 == sz);
			//
			sg.x1 = x1;
			sg.x2 = x1a;
			sg.coeff = float(h1) / float(h1 + h2);
			plan[lvl].seg_info.push_back(sg); // work to division point
			//
			init_segment(lvl + 1, x1a, x1); // backward work
			init_segment(lvl + 1, x2a, x2); // forward work
		} else{//leaf, fits as a whole
			lvl = leaf_level;
			sg.x1 = x1;
			sg.x2 = x2;
			sg.coeff = fINF; // not needed
			plan[lvl].seg_info.push_back(sg);
		};
	};
	/*
    void rebind(float * C_in, float * w0 = 0, float * w1 = 0, float * sol = 0){
        C_in._beg = C_in;
        C_out = C2.recast<warp_packf>();
        C_total = C_total.recast<warp_packf>();
        x_out = sol;
        this->C1 = C1;
        this->C2 = C2;
        this->C_total = C_total;
        this->sol = sol;
        this->pw = pw;
    };
	 */

	void init_global(){
		check_all(C1);
		check_all(C2);
		check_all(C_total);
		check_all(M);
		check_all(weights);

		global.C_in = C1.recast<warp_packf>().subdim<0>(0);
		global.C_out = C2.recast<warp_packf>().subdim<0>(0);
		global.C_total = C_total.recast<warp_packf>().subdim<0>(0);
		global.M_in = M.recast<warp_packf>().subdim<0>(0);
		global.LB_out = _LB;
		global.x_out = sol;
		global.pw = pw;
		runtime_check(weights.stride_bytes()[0] == sizeof(float)) << weights;
		global.weights = weights.subdim<0>(0);
		if(K % LABELS_PACK != 0){
			std::stringstream s;
			throw_error() << "Error: Number of labels (" << K << ")must be divisible by LABELS_PACK (" << LABELS_PACK << ") compile time define in slp_kernel.cuh.";
		};

		//std::cout << "global.C_out" << global.C_out << "\n";

		global.labels_group = K / LABELS_PACK;
		global.shfl_up_c = (WARP_SIZE - LABELS_GROUP) << 8;
		global.shfl_dwn_c = ((WARP_SIZE - LABELS_GROUP) << 8) | 0x1f;
		//cudaMemcpyToSymbol(global, &glob, sizeof(glob));
	};


	void compute_mask(){
		// simulate execution
		M_mask.resize((N + 2)*n_levels);
		//ndarray_ref<char, 2> M(&M_mask[0],N+2,1,n_levels,N+2);
		ndarray_ref<char, 2> M(&M_mask[0], intn<2>(N + 2, n_levels),ndarray_flags::host_only);
		//std::cout << M.size() << " strides: " << M.stride_bytes() << "\n";
		//std::cout << M;
		std::fill(M_mask.begin(), M_mask.end()-1,'.');
		M_lookup.resize(N);
		//M_dir.resize(N);
		std::fill(M_lookup.begin(), M_lookup.end(), -1); //will denote unused entry
		//std::fill(M_dir.begin(), M_dir.end() - 1, ' ');
		//std::fill((int*)M_mask, (int*)M_mask + 32*n_levels - 1, 0);

		char w_left = '<';
		char w_right = '>';
		char r_left = ']';
		char r_right = '[';
		char rw_left = '*';
		char rw_right = '*';

		for (int l = 0; l <= leaf_level; ++l){
			M(N, l) = '\n';
			M(N + 1, l) = '\0';
			for (int s = 0; s < (int)plan[l].seg_info.size(); ++s){// al segments in the level
				segment_info_cmem & sg = plan[l].seg_info[s];
				// fill their messages
				// will read at x1 and x2+dir and write [x1 x2] => [x1+dir x2] overwritten
				int x2 = sg.x2;
				if (l == leaf_level || l ==0 )x2 = sg.x2 - sg.dir();
				if (sg.dir()>0){
					M(sg.x1,l) = r_right;
					M(x2 + sg.dir(),l) = rw_right;
				} else{
					M(sg.x1,l) = r_left;
					M(x2 + sg.dir(),l) = rw_left;
				};
				// for leaf levels [x1 x2] spans whole chunk, they will have to recompute inner entries
				for (int x = sg.x1; x != x2;){
					x += sg.dir();
					if (sg.dir()>0){
						M(x,l) = w_right;
					} else{
						M(x,l) = w_left;
					};
				};
			};
			//std::cout << "lvl " << l << " " << M.ptr(0,l);
		};
		//cleanup
		for (int l = leaf_level-1; l>=0; --l){
			for(int x = 0; x<N;++x){
				if(M(x,l)==w_right || M(x,l)==w_left){
					for(int k = 1;k <= l; ++k){
						M(x,l-k) = '.';
					};
				};
				/*
				if(M(x,l)==r_left || M(x,l)==r_right || M(x,l)==rw_left || M(x,l)==rw_right){
					for(int k = 1;k <= l && M(x,l-k)=='.'; ++k){
						M(x,l-k) = '|';
					};
				};
				 */
			};
		};
		//std::replace(M_mask.begin(), M_mask.end(), (char)w_left, '<');
		std::string num[3];
		char s[10];
		/* printing of sime debug numbers
		for(int i=0;i<3;++i){
			num[i].resize(N);
			for(int j=0;j<N;++j){
				sprintf(s,"%3d",j);
				num[i][j] = s[i];
			};
			//std::cout << num[0]<<"\n";
		};
		for(int i=0;i<3;++i){
			pf_stream(1) << num[i]<<"\n";
		};
		*/
		//pf_stream(1) << M_mask;
		// fill segment lists
		int e_idx = 0;
		for (int l = 0; l < leaf_level; ++l){
			for (int s = 0; s < (int)plan[l].seg_info.size(); ++s){// all segments in the level
				segment_info_cmem & sg = plan[l].seg_info[s];
				// go throug positions in the segment
				int k = 0;
				runtime_check((sg.x1 < sg.x2 && sg.dir() == 1) || (sg.x1 > sg.x2 && sg.dir() == -1));
				//std::cout << sg.x1 << " " << sg.x2 <<"\n";
				for (int x = sg.x1;; x += sg.dir()){
					if (M(x,l) != '.'){// non-empty entry
						//std::cout<<"l"<<l<<" s"<<s<<" x"<<x<<" k"<<k <<"\n";
						if (k >= COUNT_OF(sg.M_list)){
							throw_error() << "Maximum segment list index is exceeded (too big image for this version or more const mem is needed)\n";
						};
						sg.M_list[k].pos = x;
						++k;
						//fill index
						if(M_lookup[x]==-1){
							M_lookup[x] = e_idx;
							++e_idx;
						};;
						if (x == sg.x2)break;
					};
				};
				/*
				if(l==0){ // zero level writes for handshake
					std::cout<<"l"<<l<<" s"<<s<<" x"<<sg.x2<<" k"<<k <<"\n";
					sg.M_list[k].pos = sg.x2;
					sg.M_list[k].idx = 0;
					++k;
				};
				 */
				// terminate with some unreachable position
				if (k >= COUNT_OF(sg.M_list)){
					throw_error() << "Maximum segment list index is exceeded (too big image for this version or more const mem is needed)\n";
				};
				sg.M_list[k].pos = -1;
			};
		};
		for(int i=0;i<3;++i){
			num[i].resize(N);
			for(int j=0;j<N;++j){
				sprintf(s,"%3d",M_lookup[j]);
				num[i][j] = s[i];
			};
			//std::cout << num[0]<<"\n";
		};
		for(int i=0;i<3;++i){
			pf_stream(1) << num[i]<<"\n";
		};
		/*
		print_array("Lookup", &M_lookup[0], N);
		for (int l = 0; l <= leaf_level; ++l){
			pf_stream(1) << "lvl" << l << "|";
			for (int x = 0; x < N; ++x){
				M_mask[l][x] = (M_lookup[x] <= l);
				char c = M_dir[x];
				if (!M_mask[l][x]){
					c = '.';
				} else;
				printf("%c ", c);
			};
			pf_stream(1) << "\n";
		};
		print_array("M_mask=\n", ndarray_ref<int, 1>((int*)M_mask, n_chains, 32), n_levels);
		 */
	};
	/*
	void init_messages(){
	dim3 dimBlock(NUM_LABELS, LABEL_GROUPS_PER_WARP*4, 1);
	dim3 dimGrid(1, divup(n_chains, dimBlock.y), 1);
	init_messages_k << <dimGrid, dimBlock >> >(0);
	init_messages_k << <dimGrid, dimBlock >> >(N-1); //asynchronous
	};
	 */
	void handshake0(int x){
		dim3 dimBlock(LABELS_GROUP, LABEL_GROUPS_PER_WARP * 2, 1); // like 2 warps per block ~ 50%
		dim3 dimGrid(1, divup(n_chains, dimBlock.y), 1);
		//handshake_k <<< dimGrid, dimBlock >>>(x, plan[0].seg_info[0].coeff);
		launch_handshake_k (dimGrid, dimBlock, x, plan[0].seg_info[0].coeff);
	};
	void work_level(int lvl){
		cudaDeviceSynchronize();
		cuda_check_error();
		if (lvl == 0){
			//init_messages();
			//cudaDeviceSynchronize();
			dim3 dimBlock(LABELS_GROUP, LABEL_GROUPS_PER_WARP * 4, 1); // like 4 warps per block ~ 100%
			dim3 dimGrid(1, divup(n_chains, dimBlock.y), 2);
			pf_stream(1) << "work init: " << (int)dimGrid.y << " x " << (int)dimGrid.z << " blocks";
			pf_stream(1) << "\t\t" << int(dimBlock.x*dimBlock.y*dimBlock.z / WARP_SIZE) << " warps/block\n";
			//work_pack <<< dimGrid, dimBlock >>>(plan[0].start, false);
			launch_work_pack (dimGrid, dimBlock, plan[0].start, false);
			cudaDeviceSynchronize();
			cuda_check_error();
			//
			assert(plan[0].seg_info.size() == 2);
			assert(plan[0].seg_info[0].x2 + 1 == plan[0].seg_info[1].x2);
			handshake0(plan[0].seg_info[0].x2);
			//
			//cudaDeviceSynchronize();
			//cuda_check_error();
			//
		} else if (lvl < leaf_level){
			//int nseg = 1 << (lvl + 1);
			//int blockz = max(nseg, 8);
			if (plan[lvl].seg_info.size() > 0){
				int n_lines = LABEL_GROUPS_PER_WARP * 4; // max occpancy for large problems
				dim3 dimBlock(LABELS_GROUP, n_lines, 1);
				dim3 dimGrid(1, divup(n_chains, dimBlock.y), (int)plan[lvl].seg_info.size());
				pf_stream(1) << "work: " << (int)dimGrid.y << " x " << (int)dimGrid.z << " blocks";
				pf_stream(1) << "\t\t" << int(dimBlock.x*dimBlock.y*dimBlock.z / WARP_SIZE) << " warps/block\n";
				//work_pack <<<  dimGrid, dimBlock >>>(plan[lvl].start, true);
				launch_work_pack(dimGrid, dimBlock , plan[lvl].start, true);
				//cudaDeviceSynchronize();
				//cuda_check_error();
			};
		} else{
			int z = 0;
			//leaf level
			int n_lines = LABEL_GROUPS_PER_WARP * 4; // max occpancy for large problems
			if(n_lines > MAX_LINES_PER_BLOCK){
				runtime_check(MAX_LINES_PER_BLOCK >= LABEL_GROUPS_PER_WARP) << "Cannot fill a single wrap\n";
				n_lines = rounddown(MAX_LINES_PER_BLOCK,LABEL_GROUPS_PER_WARP); // multiple of LABEL_GROUPS_PER_WARP, but not larger than MAX_LINES_PER_BLOCK
			};
			dim3 dimBlock(LABELS_GROUP, n_lines, 1);
			dim3 dimGrid(1, divup(n_chains, dimBlock.y), (int)plan[lvl].seg_info.size());
			runtime_check(dimBlock.y <= MAX_LINES_PER_BLOCK) << "block size " << (int)dimBlock.y << "(# lines) excceds MAX_LINES_PER_BLOCK =" <<MAX_LINES_PER_BLOCK<<"\n";

			pf_stream(1) << "work leaf: " << (int)dimGrid.y << " x " << (int)dimGrid.z << " blocks";
			pf_stream(1) << "\t\t" << int(dimBlock.x*dimBlock.y*dimBlock.z / WARP_SIZE) << " warps/block\n";
			//unrolled_sm <<< dimGrid, dimBlock >>>(plan[lvl].start);
			launch_unrolled_sm (dimGrid, dimBlock, plan[lvl].start);
			//cudaDeviceSynchronize();
		};
	};
};

/*
void slack_prop_pass(ndarray_ref<float, 2> C1, ndarray_ref<float, 2> C2, ndarray_ref<float, 2> C_total, ndarray_ref<int, 2> sol){
// number of labels is predefined

int K = NUM_LABELS;
int N = C1.size(0);
int n_chains = C1.size(1);

pw_term_device pw;
pw.L1 = 0.1;
pw.L2 = 0.3;
};
 */

////______________________________________TESTS_____________________________________________________________
//void slack_prop_test(ndarray_ref<float, 3> C1, ndarray_ref<float, 3> C2, ndarray_ref<float, 3> C_total, ndarray_ref<int, 2> sol, ndarray_ref<float, 2> weights){
//	int K = C1.size(2);
//	// number of labels is predefined
//	//int K = NUM_LABELS;
//	//int N = C1.size(0);
//	//int n_chains = C1.size(1);
//	pw_term_device pw(1.0f,3.0f,2);
//
//	//pw.L1 = 0.0f;//100.0f;
//	//pw.L2 = 0.0f;
//
//	//pw.L1 = 100.0f;
//	//pw.L2 = 100.0f;
//	//
//	slack_prop_driver alg;
//	/*
//	alg.C1 = C1;
//	alg.C2 = C2;
//	alg.C_total = C_total;
//	alg.sol = sol;
//	alg.pw = pw;
//	alg.K = K; */
//	alg.init(C1,C2,C_total,sol,pw,K,weights);
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
//	alg.execute();
//	cudaEventRecord(stop); 	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop); printf("core time: %g ms\n", milliseconds);
//};


// ______________________________________________________________________________________________

class slack_prop_2D{
public:
	ndarray_ref<float,3> C_total;
	//ndarray_ref<float,2> w0;
	//ndarray_ref<float,2> w1;
	ndarray<float, 4> W;
	//
	//hd_array<int> sol;
	ndarray_ref<int, 2> sol;
	ndarray_ref<int, 3> sols;
	//ndarray_ref<int, 2> sol_h;
	double LB;
	slack_prop_ops ops;
	ndarray<float, 3> C1;
	std::vector<hist_struct> hist;
private:
	slack_prop_driver dim_0;
	slack_prop_driver dim_1;
public:
	slack_prop_2D(){};

	int K()const{
		return dim_0.K;
	}

	slack_prop_2D(const ndarray_ref<float, 3> & C_total, ndarray_ref<int,2> sol){
		init(C_total, sol);
	}

	slack_prop_2D(ndarray_ref<float, 3> C, ndarray_ref<float, 4> W, ndarray_ref<int, 2> sol){
		init(C_total, W, sol);
	}

	void init(const ndarray_ref<float, 3> & C_total, ndarray_ref<int, 2> & sol){
		check_all(C_total);
		check_all(sol);
		int size[4] = { 2, C_total.size(1), C_total.size(2), 2}; // 2 directions, 2 channels
		W.create<memory::GPU>(size);
		init(C_total, W, sol);
	}

	void update_W(ndarray_ref<float, 2> w0, ndarray_ref<float, 2> w1){
		runtime_check(w0.size(0) == C_total.size(1) && w0.size(1) == C_total.size(2));
		runtime_check(w1.size(0) == C_total.size(1) && w1.size(1) == C_total.size(2));
		runtime_check(W.size(1) == C_total.size(1) && W.size(2) == C_total.size(2));
		check_all(w0);
		check_all(w1);
		check_all(W);
		copy_data(W.subdim<0, 3>(0, 0), w0) *= float(ops.L1);
		copy_data(W.subdim<0, 3>(1, 0), w0) *= (float(ops.L2 - ops.L1) / float(ops.delta - 1)); // slope
		copy_data(W.subdim<0, 3>(0, 1), w1) *= float(ops.L1);
		copy_data(W.subdim<0, 3>(1, 1), w1) *= (float(ops.L2 - ops.L1) / float(ops.delta - 1)); // slope
	}

	void update_W(ndarray_ref<float, 4> W){
		this->W << W;
		// convert channel 1 to slope, all directions
		float c = 1.0f/(ops.delta-1);
		madd2(W.subdim<0>(1), W.subdim<0>(1), W.subdim<0>(0), c, -c);
	}

	const ndarray_ref<int, 2> & get_sol(int s)const{
		return sols.subdim<2>(s);
	};

protected:
	//! detail -init
	void init(ndarray_ref<float, 3> C, ndarray_ref<float, 4> W, ndarray_ref<int, 2> sol){
		cuda_check_error();
		C_total = C;
		this->W = W;
		pw_term_device pw((float)ops.L1,(float)ops.L2,(int)ops.delta);
		//
		C1.create<memory::GPU>(C_total); // copy shape
		runtime_check(W.size(1) == C_total.size(1) && W.size(2) == C_total.size(2)) << W << C_total;
		dim_0.ops = ops;
		dim_1.ops = ops;
		dim_0.init(C_total, C1, C_total, sol, pw, W.subdim<3>(0));
		dim_1.init(C1.swap_dims(1,2), C1.swap_dims(1,2), C_total.swap_dims(1,2), sol.transp(), pw, W.subdim<3>(1).swap_dims(1,2));
		LB = 0;
		cuda_check_error();
	}

public:
	//void rebind(float * C_total, float * w0 = 0, float * w1 = 0){
	void rebind(float * C_total, float * W = 0){
		this->C_total.ptr() = C_total;
		if(W){
			this->W.ptr() = W;
		};
	}

	//! detail - update internal copies, before execution
	void update(){
		dim_0.C_total = this->C_total;
		dim_1.C_total = this->C_total.swap_dims(1,2);
		pw_term_device pw((float)ops.L1, (float)ops.L2, (int)ops.delta);
		dim_0.pw = pw;
		dim_1.pw = pw;
		dim_0.weights = W.subdim<3>(0);
		dim_1.weights = W.subdim<3>(1).swap_dims(1,2);
		dim_0.ops = ops;
		dim_1.ops = ops;
	}

	//! detail - solver
	void resolve_and_split(int start = 0){
		cuda_check_error();
		update();

		dim_0.C1 = C_total;
		dim_1.C1 = C_total.swap_dims(1,2);

		if(start == 0){
			for(int it=0; it < ops.total_it-1; ++it){
				dim_0.global.cost_to_return = return_types::resolve;
				dim_0.execute();
				LB = dim_0.LLB;
				if(ops.compute_LB) pf_stream(1) << "LB = "<< LB <<"\n";
				dim_1.global.cost_to_return = return_types::resolve;
				//				if(it==ops.total_it-1){
				//					dim_1.global.cost_to_return = global_info_cmem::split;
				//				};
				dim_1.execute();
				LB = dim_1.LLB;
				if(ops.compute_LB) pf_stream(1) << "LB = "<< LB <<"\n";
			};
			dim_0.global.cost_to_return = return_types::split;
			dim_0.execute();
			LB = dim_0.LLB;
			if(ops.compute_LB) pf_stream(0) << "LB = "<< LB <<"\n";
		}else{
			for(int it=0; it < ops.total_it-1; ++it){
				dim_1.global.cost_to_return = return_types::resolve;
				dim_1.execute();
				LB = dim_1.LLB;
				if(ops.compute_LB) pf_stream(1) << "LB = "<< LB <<"\n";
				dim_0.global.cost_to_return = return_types::resolve;
				//				if(it==ops.total_it-1){
				//					dim_0.global.cost_to_return = global_info_cmem::split;
				//				};
				dim_0.execute();
				LB = dim_0.LLB;
				if(ops.compute_LB) pf_stream(1) << "LB = "<< LB <<"\n";
			};
			dim_1.global.cost_to_return = return_types::split;
			dim_1.execute();
			LB = dim_1.LLB;
			if(ops.compute_LB) pf_stream(0) << "LB = "<< LB <<"\n";
		};
	};

	//! run algorithm
	void execute(){
		cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
		cuda_check_error();
		update();

		dim_0.global.cost_to_return = return_types::reminder;
		dim_0.C1 = C_total; //start from C_total
		dim_0.ops.compute_LB = false;

		//dim_1.C1 = C1.swap_dims(1,2);
		dim_1.C1 = C_total.swap_dims(1,2);
		dim_1.global.cost_to_return = return_types::reminder;

		//slack_prop_driver & D0 = ops.start==0? dim_0 : dim_1;
		//slack_prop_driver & D1 = ops.start==0? dim_1 : dim_0;

		double t1 = 0;

		for (int it = 0; it < ceil(ops.total_it); ++it){
			bool done = false;
			if(ops.compute_LB) pf_stream(0) << "it"<<it <<"\t";
			cudaEventRecord(start);
			cudaDeviceSynchronize();
			cuda_check_error();
			if(ops.debug_return){
				dim_0.global.cost_to_return = ops.cost_to_return;
			};
			if(it >= ops.total_it - 0.51){ // fractional iteration
				dim_0.global.cost_to_return = return_types::total_free;
				dim_0.ops.compute_LB = 1;
				done = true;
			};
			dim_0.execute();
			LB = dim_0.LLB;
			//print_C1();
			//break;
			if(!done){
				if(ops.debug_return){
					if(ops.cost_to_return == return_types::messages){
						C1 << dim_1.M;
					};
					done = true;
					return;
				};
			};
			if(!done){
				if (it == ceil(ops.total_it) - 1){// last iteration
					// last pass outputs free cost in C1
					dim_1.global.cost_to_return = return_types::total_free;
					// also compute the final LB
					dim_1.ops.compute_LB = 1;
					done = true;
				};
				cudaDeviceSynchronize();
				cuda_check_error();
				dim_1.C1 = C1.swap_dims(1,2);
				dim_1.execute();
				LB = dim_1.LLB;
				dim_0.C1 = C1; // continue with current cost
			};
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			if(ops.compute_LB) pf_stream(0) << "LB = "<< LB <<"\n";
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			t1 += milliseconds;
			//hist.push_back(hist_struct(LB, milliseconds*1e-3));// record in seconds
			hist.push_back(hist_struct(LB, t1*1e-3));// record in seconds
			if(done) break;
		};
		cudaDeviceSynchronize();
		//
		char s[2048];
		sprintf(s,"slack_prop time: %4.2f ms (%3.2f per iteration)\n", double(t1), double(t1/ops.total_it));
		//printf("%s",s);
		std::stringstream ss;
		ss << s;
		pf_stream(0) << ss.str();
	}
};

//_______________________________________________________________________
flow_alg::flow_alg(){
	U = 0;
	V = 0;
};
flow_alg::~flow_alg(){
	if (U) delete U;
	if (V) delete V;
};

void flow_alg::init(){
	U = new slack_prop_2D();
	V = new slack_prop_2D();
	U->ops = ops;
	V->ops = ops;
	//std::cout << "C= " << C << "\n";
	U->init(C, sol_U);
	V->init(C, sol_V);
	start = 0;
};

void flow_alg::check(){
	check_all(C);
	check_all(U->W);
	check_all(V->W);
};

void flow_alg::rebind_U(float * C){
	U->rebind(C);
};

void flow_alg::rebind_V(float * C){
	V->rebind(C);
};

void flow_alg::iterate_U(){
	pf_stream(0) << "U-iteration \n";
	U->ops = ops;
	U->update_W(w0_U, w1_U);
	U->resolve_and_split(start);
};

ndarray_ref<float, 3> flow_alg::get_hold_out_dim1_U(){
	return U->C1;
};

void flow_alg::iterate_V(){
	pf_stream(0) << "V-iteration \n";
	V->ops = ops;
	//V->resolve_and_split(start);
	V->update_W(w0_V, w1_V);
	V->resolve_and_split(!start);
	start = !start;
};

ndarray_ref<float, 3> flow_alg::get_hold_out_dim1_V(){
	return V->C1;
};

//_______________________________________________________________________

slack_prop_2D_alg::slack_prop_2D_alg(){
	alg = new slack_prop_2D();
}

slack_prop_2D_alg::~slack_prop_2D_alg(){
	delete alg;
}

void slack_prop_2D_alg::init(ndarray_ref<float, 3> C, ndarray_ref<int, 2> sol){
	runtime_check(C.stride_bytes(0) == sizeof(float)) << "C must be contiguous" << C;
	alg->ops = ops;
	alg->init(C, sol);
}

//! w0, w1 [image_size0 x image_size1]
void slack_prop_2D_alg::update_W(ndarray_ref<float, 2> w0, ndarray_ref<float, 2> w1){
	alg->ops = ops;
	alg->update_W(w0, w1);
}

//! W [channels x image_size0 x image_size1 x n_dirs]
void slack_prop_2D_alg::update_W(ndarray_ref<float, 4> W){
	alg->ops = ops;
	alg->update_W(W);
}

void slack_prop_2D_alg::solve_host_io(const ndarray_ref<float,3> & C, const ndarray_ref<float,2> & w0, const ndarray_ref<float,2> & w1, ndarray_ref<float,2> & solution){
	// copy do device
	ndarray<float,3> C_d;
	C_d.create<memory::GPU>(C);
	copy_data(C_d, C);

	ndarray<float,2> w0_d;
	w0_d.create<memory::GPU>(w0);
	copy_data(w0_d, w0);

	ndarray<float,2> w1_d;
	w1_d.create<memory::GPU>(w1);
	copy_data(w1_d, w1);

	ndarray<int,2> sol_d;
	sol_d.create<memory::GPU_managed>(solution.size());

	init(C_d, sol_d);
	update_W(w0_d,w1_d);

	alg->execute();
	// copy back to host
	solution << sol_d; // should resolve as a device function
}

/*
void slack_prop_2D_alg::rebind(float * C_total, float * W){
	alg->rebind(C_total, W);
}
*/

/*
int slack_prop_2D_alg::K()const{
	return alg->K();
}
*/

void slack_prop_2D_alg::execute(){
	alg->ops = ops;
	alg->execute();
	hist = alg->hist;
}

double slack_prop_2D_alg::LB()const{
	return alg->LB;
}

ndarray_ref<float, 3> slack_prop_2D_alg::get_C1()const{
	return alg->C1;
}

ndarray_ref<int, 2> slack_prop_2D_alg::get_sol(int s)const{
	return alg->get_sol(s);
}

//______________________________________________________________
////_________________________tests______________________________
//
//void test_1_pass(int K, int X, int Y){
//	//const int K = NUM_LABELS;
//	ndarray<float, 3> C_total;
//	C_total.create<memory::GPU_managed>(X, Y, K);
//	ndarray<float, 3> C1;
//	C1.create<memory::GPU_managed>(X, Y, K);
//	ndarray<int, 2> sol;
//	sol.create<memory::GPU_managed>(X, Y);
//	ndarray<float, 2> w0;
//	w0.create<memory::GPU_managed>(X, Y);
//	narray<float, 2> w1;
//	w1.create<memory::GPU_managed>(X, Y);
//	for (int i = 0; i < K*X*Y; ++i){
//		//C_total[i] = C1[i] = float(rand() % 10);
//		C_total.ptr()[i] = C1.ptr()[i] = float(rand() % 1000) / 1000;
//		//C_total[i] = C1[i] = i;
//		//C_total[i] = C1[i] = 1;
//	};
//	for (int i = 0; i < X*Y; ++i){
//		w0.ptr()[i] = 1;
//		w1.ptr()[i] = 1;
//	};
//	//C1.to_dev();
//	//print_array("Cost=", ndarray_ref<float, 2>(C_total._host, X, K, Y, K*X), K, 2);
//	//
//	//
//	/*
//	ndarray_ref<float, 2> C_total_s(C_total._device, X, K, Y, K*X);
//	ndarray_ref<float, 2> C1_s(C1._device, X, K, Y, K*X);
//	ndarray_ref<int, 2> sol_s(sol._device, X, 1, Y, X);
//	ndarray_ref<float, 2> w0s(w0._device, X, 1, Y, X);
//	*/
//
//	slack_prop_test(C_total, C1, C_total, sol, w0);
//
//	//sol.to_host();
//	//C1.to_host();
//	/*
//	print_array("C2=", ndarray_ref<float, 2>(C1._host, X, K, Y, K*X), K, 2);
//	print_array("sol=", ndarray_ref<int, 2>(sol._host, X, 1, Y, X), 1);
//	*/
//};
//

/*
void test_2D_pass(int K, int X, int Y){
	//const int K = NUM_LABELS;
	hd_array<float> C_total(K*X*Y);
	for (int i = 0; i < K*X*Y; ++i){
		C_total[i] = float(rand() % 10);
	};
	print_array("Cost=", ndarray_ref<float, 2>(C_total._host, X, K, Y, K*X), K, 2);
	C_total.to_dev();
	//
	ndarray_ref<float, 2> C_total_s(C_total._device, X, K, Y, K*X);
	slack_prop_2D alg(C_total_s, K);
	//
	alg.execute();
	//sol.to_host();
	//C1.to_host();
	//print_array("C2=", ndarray_ref<float, 2>(C1._host, X, K, Y, K*X), K, 2);
	print_array("sol=", ndarray_ref<int, 2>(alg.sol._host, X, 1, Y, X), 1);
};
 */

//!
void test_2D(int K, int X, int Y){
	std::cout << "test_2D(K=" << K <<",X="<<X<<",Y="<<Y<<")\n";
	ndarray<float, 3> C_total, C_total_c;
	C_total_c.create<memory::CPU>(K, X, Y);
	C_total.create<memory::GPU>(K, X, Y);
	ndarray<int, 2> sol_c ,sol;
	//
	sol_c.create<memory::CPU>(X, Y);
	sol.create<memory::GPU>(X, Y);
	//
	ndarray<float, 2> w0;
	w0.create<memory::GPU_managed>(X, Y);
	intn<2> o = w0.stride_bytes().sort_idx();
	ndarray<float, 2> w1;
	w1.create<memory::GPU_managed>(X, Y);
	//
	for (int i = 0; i < C_total.numel(); ++i){
		//C_total[i] = float(rand() % 10);
		C_total_c.ptr()[i] = float(rand() % 1000) / 1000;
	};

	copy_data(C_total, C_total_c);

	for (int i = 0; i < X; ++i){
		for (int j = 0; j < Y; ++j){
			w0(i,j) = 1;
			w1(i,j) = 1;
		};
	};
	/*
	for (int j = 0; j < w0.size(0); ++j){
		w0(5,j) = 0; // allow jump
		w0(12,j) = 0; // allow jump
	};

	for (int j = 0; j < w1.size(1); ++j){
		w1(5,j) = 0; // allow jump
		w1(12,j) = 0; // allow jump
	};
	 */

	for (int j = 0; j < w0.size(1); ++j){
		w0(5,j) = 0; // allow jump
		w0(12,j) = 0; // allow jump
	};

	for (int i = 0; i < w1.size(0); ++i){
		w1(i,5) = 0; // allow jump
		w1(i,12) = 0; // allow jump
	};

	slack_prop_2D_alg alg;

	alg.ops.L1 = 100;
	alg.ops.L2 = 100; // strong potts
	alg.ops.total_it = 5;
	alg.init(C_total, sol);
	alg.update_W(w0, w1);
	alg.execute();
	pf_stream(1) <<"LB=" << alg.LB() <<" (ref 132035.76)\n";
	if( abs(alg.LB() - 132035.76) < 0.01){
		std::cout << "TEST PASSED \n";
	}else{
		std::cout << "TEST FAILED \n";
	}
	print_array("sol=", sol, 1);
};

//
void test1(){
	std::cout << "test1 \n" <<std::endl;
	//return;
	int X = 513;//512;
	int Y = 517;//512;
	int K = 64;//64;
	//int X = 2320;//512;
	//int Y = 2000;//512;
	//int K = 32;


	/*
	int X = 16;//512;
	int Y = 16;//512;
	int K = 4;//64;
	 */
	/*
#ifdef DEBUG
	int X = 16;
	int Y = 16;
	int K = 64;
#else
	int X = 19;//512;
	int Y = 32;//512;
	int K = 4;//64;
#endif
	 */
	int seed = 1;
	srand(seed);
	//test_1_pass(K, X, Y);
	//srand(seed);
	//test_2D_pass(K, X, Y);
	srand(seed+1);
	test_2D(K, X, Y);
};

void test_stereo(){
	std::fstream ff;
	const char * fname = "../data/tsukuba.txt";
	ff.open(fname,std::fstream::in);
	if(!ff.is_open()){
		throw_error() << "file" << fname << " not found ";
	};
	int X = 100;
	int Y = 125;
	int K = 16;
	//hd_array<float> C_total(K*X*Y);
	//hd_array<float> w0(X*Y);
	//hd_array<float> w1(X*Y);
	//hd_array<int> sol(X*Y);
	ndarray<float, 3> C_total;
	C_total.create<memory::GPU>(X, Y, K);
	ndarray<int, 2> sol;
	sol.create<memory::GPU>(X, Y);
	ndarray<float, 2> w0;
	w0.create<memory::GPU_managed>(X, Y);
	ndarray<float, 2> w1;
	w1.create<memory::GPU_managed>(X, Y);
	//
	for (int i = 0; i < X*Y; ++i){
		w0.ptr()[i] = 1;
		w1.ptr()[i] = 1;
	};
	//w0.to_dev();
	//w1.to_dev();
	//
	float val;
	int i = 0;
	while (ff >> val){
		C_total.ptr()[i] = val;
		++i;
		if (ff.peek() == ',' || ff.peek() == '\n' || ff.peek() == '\r' || ff.peek()==' ')ff.ignore();
	};
	ff.close();
	pf_stream(1)<<"Number values read:"<<i<<"\n";
	if(i<X*Y*K){
		throw_error()<< "not complete";
	};
	//C_total.to_dev();
	//
	//const int size[2] = {X,Y};
	//const int stride[2] = {K,X*K};
	slack_prop_2D_alg alg;
	alg.ops.L1 = 10;
	alg.ops.L2 = 30;
	alg.ops.total_it = 50;

	//ndarray_ref<float, 2> C_total_s(C_total._device, X, K, Y, K*X);
	//ndarray_ref<float, 2> w0s(w0._device, size[0], 1, size[1], size[0]);
	//ndarray_ref<float, 2> w1s(w1._device, size[1], 1, size[0], size[1]);
	//ndarray_ref<int, 2> sol_s(sol._device, X, 1, Y, X);

	alg.init(C_total, sol);
	alg.update_W(w0, w1);
	alg.execute();
	printf("LB = %f\n",alg.LB());
	int width = X;
	int height = Y;
	//sol.to_host();
	//ndarray_ref<int, 2> sol_h(sol._host, X, 1, Y, X);
	{
		for(int i=0;i<10;++i){
			for(int j=0;j<10;++j){
				std::cout<<sol(i,j)<<" ";
			};
			std::cout<<"\n";
		};
		//
		std::fstream ff;
		ff.open("../data/disp.txt",std::fstream::out);
		for(int i=0;i<width;++i){
			for(int j=0;j<height;++j){
				int x = sol(i,j);
				ff << x;
				if(j<height-1)ff<<", ";
			};
			if(i<width-1)ff<<"\n";
		};
		ff.close();
	};
};

void solve_text_input(std::string fname){
	ndarray<float, 3> f1;
	dlm_read(f1, fname + "_f1.txt");
	ndarray<float, 3> w;
	dlm_read(w, fname + "_w.txt");

	std::cout << "f1=" << f1 << "\n";
	print_array("f1=", f1.subdim<2>(0), -1);

	int D1 = f1.size()[1];
	int D2 = f1.size()[2];
	int K = f1.size()[0];

	ndarray<int, 2> sol;
	sol.create<memory::GPU_managed>(D1,D2);
	//
	slack_prop_2D_alg alg;
	alg.ops.total_it = 20;
	alg.ops.compute_LB = 1;
	//alg.ops.debug_return = 1; //

	ndarray<float, 2> opsa;
	dlm_read(opsa, fname + "_pw.txt");
	print_array("opsa=", opsa, -1);

	alg.ops.L1 = opsa.ptr()[0];
	alg.ops.L2 = opsa.ptr()[1];
	//alg.ops.cost_to_return = return_types::messages;
	alg.ops.cost_to_return = return_types::total_free;

	std::cout << "w=" << w << "\n";
	print_array("w=", w.subdim<1,2>(0,0), -1);

	std::cout << alg.ops;

	alg.init(f1, sol);
	alg.update_W(w.subdim<2>(0), w.subdim<2>(1));
	alg.execute();

	ndarray<float, 3> C1;
	C1.create<memory::CPU>(alg.get_C1());
	C1 << alg.get_C1();
	for(int i=0; i< D1; ++i){
		for(int j=0; j< D2;++j){
			float m = 1e10;
			for(int k=0; k<K; ++k){
				m = std::min(m,C1(k,i,j));
			};
			for(int k=0; k<K; ++k){
				C1(k,i,j) -= m;
			};
		};
	};
	//print_array(C1.subdim<1>(2));
	print_array("C1=", C1.subdim<2>(0), -1);
	//print_array("C1=", C1, -1);
};
