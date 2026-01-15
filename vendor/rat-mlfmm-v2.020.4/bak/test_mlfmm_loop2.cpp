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
#include "mlfmm.hh"
#include "currentsources.hh"
#include "mgntargets.hh"
#ifdef ENABLE_CUDA_KERNELS
#include "gpukernels.hh"
#include "gpucurrentsources.hh"
#endif

// analytical field on axis of a current loop with radius R and current I as function of z
// this function is used for cross-checking the direct calculations
arma::Mat<rat::fltp> analytic_current_loop_axis(
	const arma::Mat<rat::fltp> &z, 
	const rat::fltp R, 
	const rat::fltp I){
	
	// calculate field on axis
	const arma::Mat<rat::fltp> Bz = (arma::Datum<rat::fltp>::mu_0 / (4*arma::Datum<rat::fltp>::pi))*
		((2*arma::Datum<rat::fltp>::pi*R*R*I)/arma::pow(z%z + R*R,1.5));

	// return field on axis
	return Bz;
}

// main
int main(){
	// settings
	#ifdef ENABLE_GPU
	const bool gpu_enable = true;
	#else
	const bool gpu_enable = false;
	#endif
	const rat::fltp radius = 40e-3; // radius of current loop in [m]
	const rat::fltp current = 400; // current inside the loop in [A]
	const arma::uword num_sources = 500; // number of elements in loop
	const arma::uword num_targets = 401; // number of target points on axis
	const rat::fltp zmin = -0.05; // axis minimum
	const rat::fltp zmax = 0.05; // axis maximum
	const int num_exp_min = 2;
	
	// take into account finite length of the 
	// elements when calculating the vector potential
	const bool van_lanen = true; 

	// heuristics for maximum number of expansions
	int num_exp_max = 8;
	#ifdef RAT_DOUBLE_PRECISION
		#ifdef ENABLE_CUDA_KERNELS
			#ifdef RAT_CUDA_DOUBLE_PRECISION
				num_exp_max = 10; // maximum number of expansions tested
			#else
				if(!gpu_enable)num_exp_max = 10;
			#endif
		#endif
	#endif

	// create armadillo timer
	arma::wall_clock timer;

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// tell user what this thing does
	lg->newl();
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sValidation script that compares the%s\n",KCYN,KNRM);
	lg->msg("%soutput of the MLFMM and direct calculation%s\n",KCYN,KNRM);
	lg->msg("%sfor the field of a circular current loop to%s\n",KCYN,KNRM);
	lg->msg("%sthe analytical expression for Bz.%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");	

	// create circular elements
	lg->msg(2,"%s%sGEOMETRY%s\n",KBLD,KGRN,KNRM);

	// in anti-clockwise manner (Bz should be positive)
	lg->msg(2,"Setting up current loop with %llu sources\n",num_sources);
	lg->msg("Radius: %2.2f\n",radius);
	lg->msg("Current: %2.2f\n",current);
	lg->msg(-2,"\n");
	// arma::Mat<rat::fltp> theta = arma::linspace(0,2*arma::Datum<rat::fltp>::pi,Ns+1).t();
	// arma::Mat<rat::fltp> xn = R*arma::cos(theta);
	// arma::Mat<rat::fltp> yn = R*arma::sin(theta);
	// arma::Mat<rat::fltp> zn(1,Ns+1,arma::fill::zeros);

	// coordinates of the nodes
	// arma::Mat<rat::fltp> Rn(3,Ns+1);
	// Rn.row(0) = xn; Rn.row(1) = yn; Rn.row(2) = zn;
	const arma::Row<rat::fltp> rho = arma::Row<rat::fltp>(num_sources+1,arma::fill::ones)*radius;
	const arma::Row<rat::fltp> theta = arma::linspace<arma::Row<rat::fltp> >(0,2*arma::Datum<rat::fltp>::pi,num_sources+1);
	const arma::Mat<rat::fltp> Rn = arma::join_vert(
		rho%arma::cos(theta), rho%arma::sin(theta), 
		arma::Row<rat::fltp>(num_sources+1,arma::fill::zeros));

	// create elements from consecutive nodes
	const arma::Mat<rat::fltp> dRs = arma::diff(Rn,1,1);
	const arma::Mat<rat::fltp> Rs = (Rn.tail_cols(num_sources) + Rn.head_cols(num_sources))/2;

	// create currents
	const arma::Row<rat::fltp> Is = arma::Row<rat::fltp>(num_sources,arma::fill::ones)*current;
	const arma::Row<rat::fltp> epss = 0.7*rat::cmn::Extra::vec_norm(dRs);

	// make target coordinates
	lg->msg(2,"Setting up %llu targets along z-axis\n",num_targets);
	lg->msg("Zmin: %s%2.2f%s\n",KYEL,zmin,KNRM);
	lg->msg("Zmax: %s%2.2f%s\n",KYEL,zmax,KNRM);
	const arma::Row<rat::fltp> zt = arma::linspace<arma::Row<rat::fltp> >(zmin,zmax,num_targets);
	arma::Mat<rat::fltp> Rt(3,num_targets,arma::fill::zeros); Rt.row(2) = zt;
	lg->msg(-4,"\n");

	// setup sources and targets
	// rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create(Rs,dRs,Is,epss);
	// src->set_van_Lanen(van_lanen);
	// src->set_so2ta_enable_gpu(gpu_enable);

	// setup sources and targets
	rat::fmm::ShGpuCurrentSourcesPr src = rat::fmm::GpuCurrentSources::create(Rs,dRs,Is,epss,Rt,false,true);
	src->set_van_Lanen(van_lanen);
	// src->set_so2ta_enable_gpu(gpu_enable);

	// rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt);
	// tar->set_field_type('H',3);

	// create mlfmm solver
	const rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create();
	myfmm->set_sources(src);
	myfmm->set_targets(src);

	// create settings
	rat::fmm::ShSettingsPr stngs = myfmm->settings();
	stngs->set_enable_gpu(gpu_enable);

	// compare with direct
	myfmm->calculate_direct(lg);

	// get results
	//arma::Mat<rat::fltp> Bdir = mytargets->get_field('B',3);
	const arma::Mat<rat::fltp> Bdir = src->get_field('B');

	// running mlfmm
	lg->msg(2,"%s%sANALYTIC CALCULATION%s\n",KBLD,KGRN,KNRM);

	// compare to analytical expression for current loop
	const arma::Row<rat::fltp> Bz_analytic = analytic_current_loop_axis(zt,radius,current);
	
	// compare field to analytic
	const rat::fltp accuracy_dir_analytic = arma::as_scalar(
		arma::max(100*arma::abs(Bz_analytic - Bdir.row(2))/arma::max(rat::cmn::Extra::vec_norm(Bdir)),1));
	if(accuracy_dir_analytic>0.01)rat_throw_line("direct calculation deviates from analytic");

	// report
	lg->msg("direct compare Bz with analytic: %3.5f pct\n",accuracy_dir_analytic);
	lg->msg(-2,"\n");

	// create expansion array
	const arma::Row<int> num_exp = arma::regspace<arma::Row<int> >(num_exp_min,num_exp_max);

	// allocate accuracy
	arma::Row<rat::fltp> magnetic_field_accuracy(num_exp.n_elem);

	// running mlfmm
	lg->msg(2,"%s%sRAT-MLFMM CALCULATION%s\n",KBLD,KGRN,KNRM);

	// run over number of expansions
	for(arma::uword i=0;i<num_exp.n_elem;i++){
		// inform user
		lg->msg(2,"%snumber of expansions: %s%i%s\n",KBLU,KGRN,num_exp(i),KNRM);

		// set number of expansions
		stngs->set_num_exp(num_exp(i));

		// setup mlfmm and run
		myfmm->setup(); 
		myfmm->calculate();

		// get results
		arma::Mat<rat::fltp> Bfmm = src->get_field('B');

		// compare results to direct calculation
		magnetic_field_accuracy(i) = arma::max(arma::max(100*arma::abs(Bdir - Bfmm)/arma::max(rat::cmn::Extra::vec_norm(Bdir)),1));
		const rat::fltp diff_fmm_analytic = arma::as_scalar(arma::max(100*arma::abs(Bz_analytic - Bfmm.row(2))/arma::max(rat::cmn::Extra::vec_norm(Bdir)),1));

		// print results
		lg->msg("difference (dir2fmm) in magnetic field: %s%3.5f [pct]%s\n",KYEL,magnetic_field_accuracy(i),KNRM);
		lg->msg("mlfmm compare Bz with analytic: %s%3.5f [pct]%s\n",KYEL,diff_fmm_analytic,KNRM);
		lg->msg("mlfmm setup time: %s%3.5f [s]%s\n",KYEL,myfmm->get_setup_time(),KNRM);
		lg->msg(-2,"mlfmm execution time: %s%3.5f [s]%s\n",KYEL,myfmm->get_last_calculation_time(),KNRM);
		lg->newl();

		// make sure the accuracy is at least below ten percent
		if(magnetic_field_accuracy(i)>10)rat_throw_line("magnetic field deviates more than 10 pct from direct calculation");
	}

	// return
	lg->msg(-2);

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// checking if magnetic field accuracy descending
	if(magnetic_field_accuracy.is_sorted("descend")){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("magnetic field does not converge with number of expansions");
	}
	
	// return
	lg->msg(-2,"\n");

	// return
	return 0;
}