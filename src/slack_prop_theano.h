#ifndef slack_prop_af_h
#define slack_prop_af_h

#include "slack_prop.h"
#include "error.h"

void cuda_check_error();

using namespace af;

class slack_prop_2D_theano : public slack_prop_2D_alg{
	typedef slack_prop_2D_alg parent;
	float *sol;
public:
	void init(float *d_cost_volume, float *d_w0, float *d_w1, float *d_out, int K, int D1, int D2)
	{//, af::array & sol){
//		int K = C.dims()[0];
//		int D1 = C.dims()[1];
//		int D2 = C.dims()[2];

		// save pointer for later return
		sol = d_out;

		// permute d_cost_volume

		slice<float, 2> C_s(d_cost_volume, D1, K, D2, D1*K);
		slice<float, 2> w0_s = slice<float, 2>(d_w0, D1, 1, D2, D1 * 1).transp();
		slice<float, 2> w1_s = slice<float, 2>(d_w1, D1, 1, D2, D1 * 1).transp();
		slice<int, 2> sol_s = slice<int,2>(d_out, D1, 1, D2, D1 * 1).transp();
		//slice<int, 2> sol_s(d_out, D2, D1*1, D1, 1);
		parent::init(C_s, K, w0_s, w1_s.transp(), sol_s);
	};

public: // inherited
	// void execute();
	// double LB()const;

public:
	/*
	void execute(){//float *d_cost_volume, float *d_w0, float *d_w1)
	{
		//parent::rebind(d_cost_volume, d_w0, d_w1);
		//cuda_check_error();
		parent::execute();
	};
	*/

	float *get_sol() const
	{
		return sol;
	};

	float *get_modular_LB() const
	{
		slice<float, 2> LB = parent::modular_LB();
		return LB.begin();
	};

};

void launch_simple_kernel(float *d_y, const float *d_x, const int num);

#endif
