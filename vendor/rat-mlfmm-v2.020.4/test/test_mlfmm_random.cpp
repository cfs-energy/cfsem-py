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

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "rat/common/log.hh"
#include "mlfmm.hh"
#include "currentsources.hh"
#include "mgntargets.hh"
#include "multisources.hh"


// main
int main(){
	// settings
	#ifdef ENABLE_GPU
	const bool gpu_enabled = true;
	#else
	const bool gpu_enabled = false;
	#endif
	const arma::uword Ns = 3000; // number of current sources
	const arma::uword Nt = 1200; // number of target points
	const int num_exp_min = 2; // minimal number of expansions tested
	const rat::fltp current = 1000; // element current in [A]

	// heuristics for maximum number of expansions
	int num_exp_max = 8;
	#ifdef RAT_DOUBLE_PRECISION
		#ifdef ENABLE_CUDA_KERNELS
			#ifdef RAT_CUDA_DOUBLE_PRECISION
				num_exp_max = 10; // maximum number of expansions tested
			#else
				if(!gpu_enabled)num_exp_max = 10;
			#endif
		#else
			num_exp_max = 10;
		#endif
	#endif

	// criterion for the average number of 
	// sources and targets when setting up the tree
	const arma::uword num_refine = 100;

	// use memory efficient kernels for s2m and l2t
	const bool mem_efficient = false;

	// use larger interaction list which 
	// is more accurate but also slower
	const bool large_ilist = false;

	// take into account finite length of the 
	// elements when calculating the vector potential
	const bool van_lanen = true; 

	// create logger
	const rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// tell user what this thing does
	lg->newl();
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sValidation script performing all steps%s\n",KCYN,KNRM);
	lg->msg("%sof the MLFMM on a random collection%s\n",KCYN,KNRM);
	lg->msg("%sof targets and sources, while checking%s\n",KCYN,KNRM);
	lg->msg("%sthat the accuracy improves with the number%s\n",KCYN,KNRM);
	lg->msg("%sof expansions. Note that for calculation speed%s\n",KCYN,KNRM);
	lg->msg("%sthis is the worst case scenario for the MLFMM%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");

	// set random seed
	arma::arma_rng::set_seed(1001);

	// create quasi random current source coordinates
	lg->msg(2,"%s%sGEOMETRY%s\n",KBLD,KGRN,KNRM);
	lg->msg("Setting up %s%llu%s random current sources...\n",KYEL,Ns,KNRM);
	const arma::Mat<rat::fltp> Rs = rat::cmn::Extra::random_coordinates(0, 0.1, 0.1, 1, Ns); // x,y,z,size,N
	const arma::Mat<rat::fltp> dRs = rat::cmn::Extra::random_coordinates(0, 0, 0, 2e-3, Ns); // x,y,z,size,N
	const arma::Row<rat::fltp> Is = arma::Row<rat::fltp>(Ns,arma::fill::ones)*current;
	const arma::Row<rat::fltp> epss = 0.7*rat::cmn::Extra::vec_norm(dRs);

	// create target coordinates
	lg->msg("Setting up %s%llu%s random targets...\n",KYEL,Nt,KNRM);
	const arma::Mat<rat::fltp> Rt = rat::cmn::Extra::random_coordinates(0, 0, 0, 1, Nt); // x,y,z,size,N
	lg->msg(-2,"\n");

	// create fmm
	const rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create();

	// set settimgs
	const rat::fmm::ShSettingsPr stngs = myfmm->settings();
	stngs->set_num_refine(num_refine);
	stngs->set_large_ilist(large_ilist);
	stngs->set_enable_s2t(true);
	stngs->set_enable_gpu(gpu_enabled);
	stngs->set_memory_efficient_l2t(mem_efficient);
	stngs->set_memory_efficient_s2m(mem_efficient);

	// create targets
	const rat::fmm::ShMgnTargetsPr targets = rat::fmm::MgnTargets::create(Rt);

	// create sources
	const rat::fmm::ShCurrentSourcesPr sources = 
		rat::fmm::CurrentSources::create(Rs,dRs,Is,epss);
	sources->set_van_Lanen(van_lanen);

	// add sources and targets to mlfmm
	myfmm->set_sources(sources);
	myfmm->set_targets(targets);

	// inform user
	lg->msg(2,"%s%sDIRECT CALCULATION (%s)%s\n",KBLD,KGRN,gpu_enabled ? "GPU" : "CPU",KNRM);
	lg->msg("For comparison");

	// compare with direct
	myfmm->calculate_direct();

	// get results
	const arma::Mat<rat::fltp> Adir = targets->get_field('A');
	const arma::Mat<rat::fltp> Bdir = targets->get_field('B');
	if(!Adir.is_finite())rat_throw_line("direct vector potential calculation is not finite");	
	if(!Bdir.is_finite())rat_throw_line("direct magnetic field calculation is not finite");

	// get timer in seconds
	lg->msg(-2,"time direct: %s%3.5f [s]%s\n",KYEL,myfmm->get_last_calculation_time(),KNRM);
	lg->newl();

	// inform user
	lg->msg(2,"%s%sMLFMM (%s)%s%s (num_exp = %i to %i)%s\n",KBLD,KGRN,gpu_enabled ? "GPU" : "CPU",KNRM,KGRN,num_exp_min,num_exp_max,KNRM);
	
	// create expansion array
	const arma::Row<int> num_exp = arma::regspace<arma::Row<int> >(num_exp_min,num_exp_max);

	// allocate accuracy
	arma::Row<rat::fltp> Adiffmax(num_exp.n_elem), Bdiffmax(num_exp.n_elem);

	// run over number of expansions
	for(arma::uword i=0;i<num_exp.n_elem;i++){
		// inform user
		lg->msg(2,"%snumber of expansions: %s%i%s\n",KBLU,KGRN,num_exp(i),KNRM);

		// set number of expansions
		stngs->set_num_exp(num_exp(i));

		// setup mlfmm
		myfmm->setup();

		// run multipole method
		myfmm->calculate();

		// get results
		const arma::Mat<rat::fltp> Afmm = targets->get_field('A');
		const arma::Mat<rat::fltp> Bfmm = targets->get_field('B');
		if(!Afmm.is_finite())rat_throw_line("fmm vector potential calculation is not finite");	
		if(!Bfmm.is_finite())rat_throw_line("fmm magnetic field calculation is not finite");

		// compare results to direct calculation
		const arma::Mat<rat::fltp> Adiff = Adir - Afmm;
		const arma::Mat<rat::fltp> Bdiff = Bdir - Bfmm;

		// compare
		Adiffmax(i) = arma::max(arma::max(100*arma::abs(Adiff)/arma::max(rat::cmn::Extra::vec_norm(Adir)),1));
		Bdiffmax(i) = arma::max(arma::max(100*arma::abs(Bdiff)/arma::max(rat::cmn::Extra::vec_norm(Bdir)),1));
		
		// print results
		lg->msg("difference vector potential: %s%3.5f [pct]%s\n",KYEL,Adiffmax(i),KNRM);
		lg->msg("difference magnetic field: %s%3.5f [pct]%s\n",KYEL,Bdiffmax(i),KNRM);
		lg->msg("mlfmm setup time: %s%3.5f [s]%s\n",KYEL,myfmm->get_setup_time(),KNRM);
		lg->msg("mlfmm execution time: %s%3.5f [s]%s\n",KYEL,myfmm->get_last_calculation_time(),KNRM);

		// make sure the accuracy is at least below ten percent
		if(Adiffmax(i)>10)rat_throw_line(
			"difference in vector potential between direct and mlfmm larger than 10 pct");
		if(Bdiffmax(i)>10)rat_throw_line(
			"difference in magnetic field between direct and mlfmm larger than 10 pct");
  
		// return
		lg->msg(-2,"\n");
	}

	// return
	lg->msg(-2);

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// checking if vector potential descending
	if(Adiffmax.is_sorted("descend") && Adiffmax.is_finite()){
		lg->msg("= accuracy vector potential: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy vector potential: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("vector potential does not converge with number of expansions");
	}

	// checking if vector potential descending
	if(Bdiffmax.is_sorted("descend") && Bdiffmax.is_finite()){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("magnetic field does not converge with number of expansions");
	}

	// go back
	lg->msg(-2,"\n");

	// display information
	//myfmm->display(lg);

	// end
	return 0;
}