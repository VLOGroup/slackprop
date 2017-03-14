#include <iostream>
//#include <iu/iuio.h>

#include "slack_prop.h"

/*
void test_stereo(){
	Stereo stereo;
	std::string fname_I1 = "/home/shekhovt/data/Army/frame_10_small.png";
	std::string fname_I2 = "/home/shekhovt/data/Army/frame_11_small.png";
	//
    float minDisp = -15;
    float maxDisp = 15;
    float dispStep = 0.25;
    float lambda_cost = 0.15;
    float lambda_fuse = 0.5;
    int iterations = 1000;
    float C = 200;
    float alpha = 11;
    float beta = 1.f;
    int filterSize = 3;
	//
    //
	Stereo stereo;
	iu::ImageCpu_32f_C1 *I1 = iu::imread_32f_C1(fname_I1);
	iu::ImageCpu_32f_C1 *I2 = iu::imread_32f_C1(fname_I2);
	stereo.initialize(I1->size(), minDisp, dispStep,maxDisp );
	stereo.setImages(*I1, *I2);
	stereo.compute(C,alpha, beta,lambda_cost,filterSize);
	stereo.fuse(lambda_fuse,iterations);
	saveMat("disp.pfm",stereo.getOutput());
	delete I1;
	delete I2;
};
*/

int main(int argc, char *argv[]){
	std::cout << "main"<<std::endl;
    //iu::ImageGpu_8u_C1 * a = iu::imread_cu8u_C1("/home/shekhovt/data/Army/frame_10_small.png");
    //iu::ImageGpu_8u_C1 * b = iu::imread_cu8u_C1("/home/shekhovt/data/Army/frame_11_small.png");
    //std::cout << "DFGDFG\n";
    //test(a);
    //test_iterative();
	//test_slack_prop2D();
	test1();
	//solve_text_input("../data/test_chain");
	//test_stereo();
	//test_stereo();
    //delete a;
    return 0;
}
