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
#include <complex>
#include <cassert>

#include "common/extra.hh"
#include "magnetictargets.hh"
#include "mgnmesh.hh"
#include "mlfmm.hh"
#include "common/log.hh"

// analytical field on axis of a radially magnetised finite cylinder
// Yoshihisa Iwashita, "Axial Magnetic field produced by radially 
// magnetized permanent magnet ring", Proceedings of the 1994 
// International Linac Conference, Tsukuba, Japan
arma::Row<double> analytic_radially_magnetised_ring_axis(
	const arma::Row<double> z, const double M, const double Rin, 
	const double Rout, const double height){

	// calculate helper variables
	arma::Row<double> r0 = arma::sqrt(1+arma::pow(z/Rout,2));
	arma::Row<double> b0 = arma::sqrt(1+arma::pow(z/Rin,2));
	arma::Row<double> r1 = arma::sqrt(1+arma::pow((z+height)/Rout,2));
	arma::Row<double> b1 = arma::sqrt(1+arma::pow((z+height)/Rin,2));

	// calculate field
	arma::Row<double> Bz = -(arma::datum::mu_0*M/2)*(1/r1-1/b1-1/r0+
		1/b0+arma::log((1+r0)%(1+b1)/((1+b0)%(1+r1))));
	
	// return calculate value
	return Bz;
}

// main
int main(){
	// settings
	arma::uword num_exp = 8;
	arma::uword num_refine = 50;
	//arma::uword num_gauss = 4;

	// geometry
	double Rin = 0.1;
	double Rout = 0.12;
	double height = 0.02;
	arma::uword nr = 5;
	arma::uword nh = 5;
	arma::uword nl = 180;
	double M = 1e6;
		
	// create logger
	ShLogPr lg = Log::create();

	// create armadillo timer
	arma::wall_clock timer;

	// tell user what this thing does
	lg->newl();
	lg->msg("%sValidation script checking%s\n",KCYN,KNRM);
	lg->msg("%sthe field on the axis of a%s\n",KCYN,KNRM);
	lg->msg("%sradially magnetised cylinder%s\n",KCYN,KNRM);
	lg->msg("%swith analytical expressions.%s\n",KCYN,KNRM);
	lg->newl();

	// start setup 
	lg->msg("%sSetting up Geometry%s\n",KCYN,KNRM);
	lg->msg("%sRing with inner radius %.2f%s\n",KCYN,Rin,KNRM);
	lg->msg("%souter radius %.2f and height %.2f%s\n",KCYN,Rout,height,KNRM);
	lg->msg("%swith magnetisation %.2e [MA/m]%s\n",KCYN,M/1e6,KNRM);
	lg->newl();

	// create magnetised mesh
	ShMgnMeshPr mysources = MgnMesh::create();
	mysources->setup_cylinder(Rin,Rout,height,nr,nh,nl);

	// finalise setup by setting up volumes
	mysources->calculate_element_volume();
		
	// update magnetisation
	arma::Mat<double> Rn = mysources->get_node_coords();
	arma::Mat<double> Mn = M*(Rn.each_row()/Extra::vec_norm(Rn));
	mysources->set_magnetisation_nodes(Mn);

	// target points
	arma::Mat<double> Rt(3,1000,arma::fill::zeros);
	Rt.row(2) = arma::linspace<arma::Row<double> >(-0.2,0.4,Rt.n_cols);

	// create target level
	ShMagneticTargetsPr mytargets = MagneticTargets::create();
	mytargets->set_coords(Rt);

	// create settings
	ShSettingsPr settings = Settings::create();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// start setup 
	lg->msg(2,"%sCalculating with MLFMM%s\n",KCYN,KNRM);

	// setup and run MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings,mysources,mytargets);
	myfmm->set_lg(lg);

	// run mlfmm
	timer.tic();
	myfmm->setup(); 
	double tsetup = timer.toc();
	timer.tic();
	myfmm->calculate();	// report
	double tfmm = timer.toc();

	// start setup 
	lg->msg(-2,"%sCalculation Finished%s\n",KCYN,KNRM);
	lg->newl();

	// get results
	arma::Mat<double> Bfmm = mytargets->get_field("B",3);	
	arma::Row<double> Bzfmm = Bfmm.row(2);

	// analytical magnetised cylinder
	arma::Row<double> Bz = analytic_radially_magnetised_ring_axis(
		Rt.row(2)-height/2, M, Rin, Rout, height);

	// calculate difference between analytica and mlfmm
	arma::Row<double> diff = 100*arma::abs(Bfmm.row(2) - Bz)/(arma::max(Bz)-arma::min(Bz));

	// report result
	lg->msg("%sResults%s\n",KCYN,KNRM);
	lg->msg("%s= difference with analytic: %s%2.2f [pct]%s\n",KCYN,KYEL,arma::max(diff),KNRM);
	if(arma::all(diff<2)){
		lg->msg("%s= accuracy is: %sOK%s\n",KCYN,KGRN,arma::max(diff),KNRM);
	}else{
		lg->msg("%s= accuracy is: %sNOT OK%s\n",KCYN,KRED,arma::max(diff),KNRM);
	}
	lg->msg("%s= mlfmm setup time: %s%2.2f%s\n",KCYN,KYEL,tsetup,KNRM);
	lg->msg("%s= mlfmm time: %s%2.2f%s\n",KCYN,KYEL,tfmm,KNRM);
	lg->newl();

	// check result
	assert(arma::all(diff<2));

	// return
	return 0;
}