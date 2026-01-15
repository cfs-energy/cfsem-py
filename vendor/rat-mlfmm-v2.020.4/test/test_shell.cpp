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
#include <complex>

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "mgntargets.hh"
#include "currentsurface.hh"
#include "mlfmm.hh"
#include "settings.hh"
#include "currentmesh.hh"
#include "rat/common/log.hh"

// analytical equation
arma::Row<rat::fltp> analytic_shell_axis(
	const arma::Row<rat::fltp> &z, const rat::fltp R, const rat::fltp L, const rat::fltp I){

	// analytical equation
	arma::Row<rat::fltp> Bz = (arma::Datum<rat::fltp>::mu_0*I/(2*L))*(
		z/arma::sqrt(R*R + z%z) + (L-z)/arma::sqrt(R*R+(L-z)%(L-z)));

	// return calculate value
	return Bz;
}



// main
int main(){
	// settings
	const arma::uword num_exp = 6;
	const arma::uword num_refine = 100;
	const arma::uword num_targets = 1000;
	const rat::fltp J = 400e3;
	const rat::fltp R = 0.03;
	const rat::fltp h = 0.03;
	const rat::fltp dl = 2e-3;
	const arma::uword nh = std::max(2,(int)std::ceil(h/dl));
	const arma::uword nazy = std::max(2,(int)std::ceil(2*arma::Datum<rat::fltp>::pi*R/dl));

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// tell user what this thing does
	lg->newl();
	lg->msg("%sValidation script checking%s\n",KCYN,KNRM);
	lg->msg("%sthe field on the axis of a%s\n",KCYN,KNRM);
	lg->msg("%scylindrical current shell%s\n",KCYN,KNRM);
	lg->msg("%swith analytical expressions.%s\n",KCYN,KNRM);
	lg->newl();
	
	// start setup 
	lg->msg(2,"%sSetting up Geometry%s\n",KBLU,KNRM);
	lg->msg("cylinder shell with radius %s%.2f [m]%s\n",KYEL,R,KNRM);
	lg->msg("and height %s%.2f [m]%s\n",KYEL,h,KNRM);
	lg->msg("with surface current density %s%.2e [A/mm]%s\n",KYEL,1e-3*J,KNRM);
	lg->msg(-2,"\n");

	// make volume sources
	rat::fmm::ShCurrentSurfacePr mysources = rat::fmm::CurrentSurface::create();
	mysources->setup_cylinder_shell(R,h,nh,nazy,J);

	// create target level
	const arma::Row<rat::fltp> z = arma::linspace<arma::Row<rat::fltp> >(-0.1,0.2,num_targets);
	arma::Mat<rat::fltp> Rt(3,num_targets,arma::fill::zeros); Rt.row(2) = z;
	rat::fmm::ShTargetPointsPr mytargets = rat::fmm::MgnTargets::create(Rt);
	
	// output
	lg->msg("%sRunning MLFMM%s\n",KBLU,KNRM);
	lg->newl();

	// setup and run MLFMM
	const rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(mysources,mytargets);

	// input settings
	rat::fmm::ShSettingsPr stngs = myfmm->settings();
	stngs->set_num_exp(num_exp);
	stngs->set_num_refine(num_refine);

	// run calculation
	myfmm->setup(); myfmm->calculate();	// report

	// get results
	const arma::Mat<rat::fltp> Bfmm = mytargets->get_field('B');
	const arma::Row<rat::fltp> Bz1 = Bfmm.row(2);
	const arma::Row<rat::fltp> Bz2 = analytic_shell_axis(z+h/2,R,h,J*h);

	// calculate difference between analytic and mlfmm
	const arma::Row<rat::fltp> diff = 100*arma::abs(Bz1 - Bz2)/(arma::max(Bz2)-arma::min(Bz1));

	// report result
	lg->msg("%sResults%s\n",KCYN,KNRM);
	lg->msg("%s= difference with analytic: %2.2f pct%s\n",KCYN,arma::max(diff),KNRM);

	// checking if vector potential descending
	if(arma::max(diff)<2){
		lg->msg("%s= accuracy magnetic field: %sOK%s\n",KCYN,KGRN,KNRM);
	}else{
		lg->msg("%s= accuracy magnetic field: %sNOT OK%s\n",KCYN,KRED,KNRM);
		rat_throw_line("difference in magnetic field exceeds tolerance");
	}
	lg->newl();

	// return
	return 0;
}