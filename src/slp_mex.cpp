#include <iu/ndarray/mex_io.h>

#include "slack_prop.h"

struct mexFunction_loaded{
	~mexFunction_loaded(){
		mexPrintf("unloading slp_mex\n");
	};
};

mexFunction_loaded loaded;

void mexFunction_protect(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	[x hist] = mexFunction(sz,f1,w,options)
	Input 1:
	0) f1 -- [K x D1 x D2] float -- unary costs
	1) w -- [D1 x D2 x nS] float -- pairwise weights
	2) options struct double -- get available fields from an output
	Output:
	x - integer labeling
	hist - history of the lower bound, etc
	*/

	slack_prop_ops ops;
	ops["compute_LB"] = true;

	if (nrhs < 2){
		mexPrintf("Usage: \n");
		mexPrintf(
				"[x hist] = slp_mex(sz,f1,w,[options])\n"
				"Input 1:\n"
				"  f1 -- [K x D1 x D2] float -- unary costs\n"
				"  w -- [D1 x D2 x nS] float -- pairwise weights (expecting nS=2 -- edge directions)\n"
				"  options -- struct double\n"
				"Output:\n"
				"  x -- [D1 x D2] integer labeling\n"
				"  hist -- [3 x ops.total_it] history over iterations: [reserved LB time]\n"
				"  minorant -- [K x D1 x D2] -- modular minorant (lower bound) of the whole problem\n");
		mexPrintf("Default ");
		std::cout << ops << "\n" << std::flush;
		return;
	};

	ndarray_ref<float, 3> f1(prhs[0]);
	ndarray_ref<float, 3> w(prhs[1]);

	if (nrhs >= 3){
		ops << mx_struct(prhs[2]);
	};
	//
	int K = f1.size(0);
	//int nV = f1.size()[1];
	//int sz0 = ops["sz0"];
	//int sz1 = ops["sz1"];
	int sz0 = f1.size(1);
	int sz1 = f1.size(2);
	//
	ndarray<int, 2> x;
	x.create<memory::GPU>(intn<2>(sz0,sz1));
	ndarray<double, 2> hist;
	//
	// move data to GPU
	ndarray<float, 3> C;
	C.create<memory::GPU>(f1.size());
	C << f1;
	//
	ndarray<float, 3> wg;
	wg.create<memory::GPU>(w.size());
	wg << w;
	//
	{
		slack_prop_2D_alg alg;
		alg.ops << ops;
		alg.init(C,x);
		alg.update_W(wg.subdim<2>(0), wg.subdim<2>(1));
		//
		alg.execute();
		//
		hist.create<memory::CPU>({3,(int)alg.hist.size()});
		for(int i=0; i<alg.hist.size();++i){
			hist(0,i) = 0;
			hist(1,i) = alg.hist[i].LB;
			hist(2,i) = alg.hist[i].t;
		};
		//
		if (nlhs >= 1){
			plhs[0] << x;
		};

		if (nlhs >= 2){
			plhs[1] << hist;
		};

		if (nlhs >= 3){
			plhs[2] << alg.get_C1();
		};
	};
	//
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	//mexargs::MexLogStream log("log/output.txt", false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log", true);
	//debug::errstream.attach(&err);

	try{
		mexFunction_protect(nlhs, plhs, nrhs, prhs);
	} catch (std::exception & e){
		//debug::stream << "!Exception:"<<e.what()<<"\n";
		mexErrMsgTxt(e.what());
	};

	//memserver::get_global()->clean_garbage();
	//debug::stream << "mem at exit: " << memserver::get_global()->mem_used() << " bytes\n";
	//debug::stream << "end of output.\n";
	//debug::stream.detach();
	//debug::errstream.detach();
};
