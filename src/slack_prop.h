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

#include "slack_prop.kernel.h"
#include <stdio.h>
#include <assert.h>
#include <iu/ndarray/options.h>
#include <iu/ndarray/ndarray_ref.host.h>
#include <vector>


class slack_prop_ops : public options{
public:
	NEWOPTION(double, L1, 1.0f) // value at displacement 1
	NEWOPTION(double, L2, 3.0f) // value at displacement delta
	NEWOPTION(int, delta, 2)    // if = 2 - standard Hirschmuller L1-L2 model, if > 2 - truncated TV model with value at 1 still L1 (need not be convex), slope = (L2 - L1)/(delta - 1)
	NEWOPTION(double, total_it, 5)
	//NEWOPTION(int, start, 0) // subproblem starting the iterations dim0 is hirizontal chains dim1 - vertical
	NEWOPTION(bool, compute_LB, false)
	NEWOPTION(return_types, cost_to_return, return_types::total_free) // reminder, total_withheld, total_free, split, recombine, resolve
	NEWOPTION(bool, debug_return, 0) // reminder, total_withheld, total_free, split, recombine, resolve
	NEWOPTION(int, debug_terminate_level, -1) // return after a specific level in the hierarchical processing, -1 is full operation
};

class slack_prop_2D;

//______________________stereo_alg_______________________________
struct hist_struct{
	double LB;
	double t;
	hist_struct(double _LB, double _t):LB(_LB),t(_t){};
};

//______________________algorithm interface class________________
class slack_prop_2D_alg{
protected:
	slack_prop_2D * alg;
public:
	slack_prop_ops ops;
	ndarray_ref<float,4> W;
	std::vector<hist_struct> hist;
	ndarray_ref<int,3> sols;
public:
	slack_prop_2D_alg();
	~slack_prop_2D_alg();
	// init: set cost and sol, C [Depth x image_size0 x image_size1]; sol [image_size0 x image_size1 ]
	void init(ndarray_ref<float, 3> C, ndarray_ref<int,2> sol);
	//! w0, w1 [image_size0 x image_size1]
	void update_W(ndarray_ref<float, 2> w0, ndarray_ref<float, 2> w1);
	//! W [channels x image_size0 x image_size1 x n_dirs]
	void update_W(ndarray_ref<float, 4> W);
	//! depricated function to update costs and weights
	void rebind(float * C_total, float * W = 0) = delete;
	void execute();
	int K()const;
	double LB()const;
	//! obtain the output cost volume C1, the type of output is controlled by options.cost_to_return
	ndarray_ref<float, 3> get_C1()const;
	//! obtain solutions of sub-problems, s = {0,1}
	ndarray_ref<int, 2> get_sol(int s)const;
	void solve_host_io(const ndarray_ref<float,3> & C, const ndarray_ref<float,2> & w0, const ndarray_ref<float,2> & w1, ndarray_ref<float,2> & solution);
private:
	slack_prop_2D_alg(const slack_prop_2D_alg & b){};// don't copy
};

//______________________flow_alg_______________________________

class flow_alg{
public:
	slack_prop_2D * U, *V;
	int start;
public: // inputs
	ndarray_ref<float, 3> C;
	int K;
	ndarray_ref<float, 2> w0_U;
	ndarray_ref<float, 2> w1_U;
	ndarray_ref<int, 2> sol_U;
	ndarray_ref<float, 2> w0_V;
	ndarray_ref<float, 2> w1_V;
	ndarray_ref<int, 2> sol_V;
	slack_prop_ops ops;
public:
	void init();
	void check();
	void iterate_U();
	void rebind_U(float * C);
	void rebind_V(float * C);
	ndarray_ref<float, 3> get_hold_out_dim1_U();
	void iterate_V();
	ndarray_ref<float, 3> get_hold_out_dim1_V();
	flow_alg();
	~flow_alg();
};

//______________________tests_______________________________

void test1();
void test_stereo();
void solve_text_input(std::string fname);
