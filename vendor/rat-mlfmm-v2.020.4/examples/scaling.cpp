// Copyright 2025 Jeroen van Nugteren

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>
#include <cassert>

// specific headers
#include "mlfmm.hh"
#include "rat/common/extra.hh"
#include "currentsources.hh"
#include "mgntargets.hh"
#include "rat/common/log.hh"

// main
int main(){
	// settings
	const bool gpu_enabled = true;
	const bool save_timings = true;
	const bool use_pthreads = true;
	arma::Row<arma::uword> Ns = 
		arma::conv_to<arma::Row<arma::uword> >::from(
		arma::logspace<arma::Row<rat::fltp> >(1,7,30));
	const bool use_split = true;
	const int num_exp = 5;
	
	arma::Row<arma::uword> Nt = Ns;
	
	rat::fltp num_refine = RAT_CONST(256.0);

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// tell user what this thing does
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	std::printf("Script for comparing scaling with number\n");
	std::printf("of elements for both direct and MLFMM calculations.\n");
	lg->msg(-2,"\n");

	// use armadillo timer
	arma::wall_clock timer;

	// set random seed
	arma::arma_rng::set_seed(1001);

	// allocate timings
	assert(Ns.n_elem==Nt.n_elem);
	arma::Row<rat::fltp> time_direct(1,Ns.n_elem,arma::fill::zeros);
	arma::Row<rat::fltp> time_tree(1,Ns.n_elem,arma::fill::zeros);
	arma::Row<rat::fltp> time_fmm(1,Ns.n_elem,arma::fill::zeros);
	arma::Row<rat::fltp> Adiffmax(1,Ns.n_elem,arma::fill::zeros);
	arma::Row<rat::fltp> Bdiffmax(1,Ns.n_elem,arma::fill::zeros);

	// walk over number of elements
	for(arma::uword i=0;i<Ns.n_elem;i++){

		// get number of sources and targets	
		lg->msg(2,"%s%sGEOMETRY%s\n",KBLD,KGRN,KNRM);
		lg->msg("%llu random current sources\n",Ns(i));
		lg->msg("%llu random target points\n",Nt(i));
		lg->msg(-2,"\n");

		// create quasi random current source coordinates
		arma::Mat<rat::fltp> Rs = 
			rat::cmn::Extra::random_coordinates(0, 0.1, 0.1, 1, Ns(i)); // x,y,z,size,N
		arma::Mat<rat::fltp> dRs = 
			rat::cmn::Extra::random_coordinates(0, 0, 0, 2e-3, Ns(i)); // x,y,z,size,N
		arma::Row<rat::fltp> Is(Ns(i)); Is.fill(1000);
		arma::Row<rat::fltp> eps = 0.7*rat::cmn::Extra::vec_norm(dRs);

		// create source currents
		rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create(Rs,dRs,Is,eps);

		// create target coordinates
		arma::Mat<rat::fltp> Rt = 
			rat::cmn::Extra::random_coordinates(0, 0, 0, 1, Nt(i)); // x,y,z,size,N
			
		// create target level
		rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt);
		tar->set_field_type("AB",arma::Row<arma::uword>{3,3});

		// setup and run MLFMM
		rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(src,tar);

		// add settings
		rat::fmm::ShSettingsPr stngs = myfmm->settings();
		stngs->set_num_exp(num_exp);
		stngs->set_num_refine(num_refine);
		stngs->set_parallel_m2m(use_pthreads);
		stngs->set_parallel_m2l(use_pthreads);
		stngs->set_parallel_l2l(use_pthreads);
		stngs->set_split_s2t(use_split);
		stngs->set_split_m2l(use_split);
		stngs->set_parallel_l2t(use_pthreads);
		stngs->set_parallel_s2m(use_pthreads);
		stngs->set_parallel_s2t(use_pthreads);
		stngs->set_enable_gpu(gpu_enabled);

		// setup timer
		timer.tic();

		// run mlfmm
		myfmm->setup(lg); 

		// setup time
		time_tree(i) = timer.toc();

		// calculation timer
		timer.tic();

		// calculate
		myfmm->calculate(lg);	// report

		// get timer in seconds
		time_fmm(i) = timer.toc();

		// get results
		const arma::Mat<rat::fltp> Afmm = tar->get_field('A');
		const arma::Mat<rat::fltp> Bfmm = tar->get_field('B');

		// setup timer
		timer.tic();
		if(Ns(i)<200000 && Nt(i)<200000){
			// calculate direct
			myfmm->calculate_direct(lg);

			// setup time
			time_direct(i) = timer.toc();

			// get results
			const arma::Mat<rat::fltp> Adir = tar->get_field('A');
			const arma::Mat<rat::fltp> Bdir = tar->get_field('B');
		}
	}

	// display results
	lg->msg(2,"%s%sRESULTS%s\n",KBLD,KGRN,KNRM);

	// table with computing times
	lg->msg(2,"%sComputing Times%s\n",KBLU,KNRM);
	lg->msg("%s%8s %8s %8s %8s %8s%s\n",KBLD,"#src","#tar","dir [s]","tree [s]","fmm [s]",KNRM);
	for(arma::uword i=0;i<Ns.n_cols;i++){
		lg->msg("%08llu %08llu %08.3f %08.3f %08.3f\n",Ns(i),Nt(i),time_direct(i),time_tree(i),time_fmm(i));
	}
	lg->msg(-2,"\n");

	// done
	lg->msg(-2,"\n");

	if(save_timings){
		// create data
		arma::Mat<rat::fltp> M(Ns.n_elem, 5);
		M.col(0) = arma::conv_to<arma::Col<rat::fltp> >::from(Ns);
		M.col(1) = arma::conv_to<arma::Col<rat::fltp> >::from(Nt);
		M.col(2) = time_direct; M.col(3) = time_tree; M.col(4) = time_fmm;

		// create header
		arma::field<std::string> header(5);
		header(0) = "num_sources [s]";
		header(1) = "num_targets [s]";
		header(2) = "time_direct [s]";
		header(3) = "time_tree [s]";
		header(4) = "time_fmm [s]";

		// // write
		// if(use_gpu)M.save(arma::csv_name("timings_gpu.csv", header));
		// else M.save(arma::csv_name("timings_cpu.csv", header));
	}

	// return
	return 0;
}