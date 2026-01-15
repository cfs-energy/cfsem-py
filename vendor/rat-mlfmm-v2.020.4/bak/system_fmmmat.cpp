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

#include <armadillo>
#include <iostream>
#include <cmath>
#include <cassert>

#include "common/extra.hh"
#include "currentsources.hh"
#include "targets.hh"
#include "magnetictargets.hh"
#include "mlfmm.hh"
#include "settings.hh"

// analytical field on axis of a current loop with radius R and current I as function of z
// this function is used for cross-checking the direct calculations
arma::Mat<double> analytic_current_loop_axis(
	const arma::Mat<double> z, 
	const double R, 
	const double I){
	
	// calculate field on axis
	arma::Mat<double> Bz = (arma::datum::mu_0 / (4*arma::datum::pi))*
		((2*arma::datum::pi*R*R*I)/arma::pow(z%z + R*R,1.5));

	// return field on axis
	return Bz;
}

// main
int main(){
	// settings
	double R = 40e-3; // radius of current loop in [m]
	double I = 400; // current inside the loop in [A]
	arma::uword Ns = 500; // number of elements in loop
	arma::uword Nt = 50; // number of target points on axis
	double zmin = -0.05; // axis minimum
	double zmax = 0.05; // axis maximum
	arma::uword num_exp_min = 2;
	arma::uword num_exp_max = 5;

	// create logger
	ShLogPr lg = Log::create();	

	// create armadillo timer
	arma::wall_clock timer;

	// tell user what this thing does
	lg->newl();
	lg->msg("%sValidation script that compares the%s\n",KCYN,KNRM);
	lg->msg("%soutput of the MLFMM (matrix version) and%s\n",KCYN,KNRM);
	lg->msg("%sdirect calculation for the field of a%s\n",KCYN,KNRM);
	lg->msg("%scircular current loop to the analytical%s\n",KCYN,KNRM);
	lg->msg("%sexpression for Bz.%s\n",KCYN,KNRM);
	lg->newl();

	// create circular elements
	lg->msg(2,"%sSetup problem%s\n",KCYN,KNRM);

	// in anti-clockwise manner (Bz should be positive)
	lg->msg(2,"Setting up current loop with %s%llu%s sources ...\n",KYEL,Ns,KNRM);
	lg->msg("Radius: %s%2.2f%s\n",KYEL,R,KNRM);
	lg->msg(-2,"Current: %s%2.2f%s\n",KYEL,I,KNRM);
	lg->newl();
	arma::Mat<double> theta = arma::linspace(0,2*arma::datum::pi,Ns+1).t();
	arma::Mat<double> xn = R*arma::cos(theta);
	arma::Mat<double> yn = R*arma::sin(theta);
	arma::Mat<double> zn(1,Ns+1,arma::fill::zeros);

	// coordinates of the nodes
	arma::Mat<double> Rn(3,Ns+1);
	Rn.row(0) = xn; Rn.row(1) = yn; Rn.row(2) = zn;

 	// create elements from consecutive nodes
 	arma::Mat<double> dRs = arma::diff(Rn,1,1);
 	arma::Mat<double> Rs = (Rn.tail_cols(Rn.n_cols-1) + Rn.head_cols(Rn.n_cols-1))/2;

 	// create currnets
 	arma::Row<double> Is(1,Ns); Is.fill(I);

	// create a set of point sources
	ShCurrentSourcesPr mysources = CurrentSources::create();
	mysources->set_coords(Rs);
	mysources->set_currents(Is%dRs.each_row());

	// make target coordinates
	lg->msg(2,"Setting up %s%llu%s targets along z-axis ...\n",KYEL,Nt,KNRM);
	lg->msg("Zmin: %s%2.2f%s\n",KYEL,zmin,KNRM);
	lg->msg(-2,"Zmax: %s%2.2f%s\n",KYEL,zmax,KNRM);
	lg->newl();
	arma::Mat<double> zt = arma::linspace(zmin,zmax,Nt).t();
	arma::Mat<double> Rt(3,Nt,arma::fill::zeros); 
	Rt.row(2) = zt;

	// create target level
	ShMagneticTargetsPr mytargets = MagneticTargets::create();
	mytargets->set_coords(Rt);

	lg->msg(-2,"setup done\n");
	lg->newl();

	// compare to analytical expression for current loop
	lg->msg("%sCalculating Bz analytically%s\n",KCYN,KNRM); 
	arma::Row<double> Bz_analytic = analytic_current_loop_axis(zt,R,I);

	// create settings
	ShSettingsPr settings = Settings::create();

	// setup and run MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings,mysources,mytargets);

	// inform user
	lg->msg(2,"%sRunning direct calculation for comparison%s\n",KCYN,KNRM);	

	// compare with direct
	timer.tic();
	myfmm->calculate_direct();
	double tdirect = timer.toc();

	// get results
	arma::Mat<double> Bdir = mytargets->get_field("B",3);
	
	// compare field to analytic
	double accuracy_dir_analytic = arma::as_scalar(arma::max(100*arma::abs(Bz_analytic - Bdir.row(2))/arma::max(Extra::vec_norm(Bdir)),1));
	assert(accuracy_dir_analytic<0.01); // depends on number of sources

	// report
	lg->msg("time direct %s%3.5f%s [s]\n",KYEL,tdirect,KNRM);
	lg->msg(-2,"direct compare Bz with analytic: %s%3.5f%s pct\n",KYEL,accuracy_dir_analytic,KNRM);
	lg->newl();

	// create expansion array
	arma::Row<arma::uword> num_exp = 
		arma::regspace<arma::Row<arma::uword> >(num_exp_min,num_exp_max);

	// allocate accuracy
	arma::Row<double> magnetic_field_accuracy(num_exp.n_elem);

	lg->msg(2,"%sRunning MLFMM%s\n",KCYN,KNRM);

	// run over number of expansions
	for(arma::uword i=0;i<num_exp.n_elem;i++){
		// inform user
		lg->msg(2,"%snumber of expansions: %s%llu%s\n",KBLU,KGRN,num_exp(i),KNRM);

		// setup matrices
		settings->set_num_exp(num_exp(i));
		settings->set_use_matrix_version(true);

		// mlfmm
	    timer.tic();
	    myfmm->setup();
	    double tsetup = timer.toc();
	    timer.tic();
	    myfmm->calculate(); 
	    double tfmm = timer.toc();
		
		// get results
		arma::Mat<double> Afmm = mytargets->get_field("A",3);
		arma::Mat<double> Bfmm = mytargets->get_field("B",3);

		// compare results to direct calculation
		magnetic_field_accuracy(i) = arma::max(arma::max(100*arma::abs(Bdir - Bfmm)/arma::max(Extra::vec_norm(Bdir)),1));
		double diff_fmm_analytic = arma::as_scalar(arma::max(100*arma::abs(Bz_analytic - Bfmm.row(2))/arma::max(Extra::vec_norm(Bdir)),1));

		// print results
		lg->msg("difference (dir2fmm) in magnetic field: %s%3.5f%s pct\n",KYEL,magnetic_field_accuracy(i),KNRM);
		lg->msg("mlfmm compare Bz with analytic: %s%3.5f%s pct\n",KYEL,diff_fmm_analytic,KNRM);
		lg->msg("mlfmm setup time: %s%3.5f%s [s]\n",KYEL,tsetup,KNRM);
		lg->msg(-2,"mlfmm execution time: %s%3.5f%s: [s]\n",KYEL,tfmm,KNRM);
		lg->newl();

		// make sure the accuracy is at least below ten percent
		assert(magnetic_field_accuracy(i)<10);
	}

	// return
	lg->msg(-2);

	// display results
	lg->msg("%sResults%s\n",KCYN,KNRM);	

	// checking if magnetic field accuracy descending
	if(magnetic_field_accuracy.is_sorted("descend")){
		lg->msg("%s= accuracy magnetic field: %sOK%s\n",KCYN,KGRN,KNRM);
	}else{
		lg->msg("%s= accuracy magnetic field: %sNOT OK%s\n",KCYN,KRED,KNRM);
	}

	// assert that the accuracy is increasing with the number of expansions
	assert(magnetic_field_accuracy.is_sorted("descend"));  

	// return
	return 0;
}