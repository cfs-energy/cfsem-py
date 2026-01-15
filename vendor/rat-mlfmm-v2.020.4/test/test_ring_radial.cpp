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
#include "momentsources.hh"
#include "mlfmm.hh"

// analytical field on axis of a radially magnetised finite cylinder
// Yoshihisa Iwashita, "Axial Magnetic field produced by radially 
// magnetized permanent magnet ring", Proceedings of the 1994 
// International Linac Conference, Tsukuba, Japan
arma::Row<rat::fltp> analytic_radially_magnetised_ring_axis(
	const arma::Row<rat::fltp> &z, const rat::fltp M, const rat::fltp Rin, 
	const rat::fltp Rout, const rat::fltp height){

	// calculate helper variables
	arma::Row<rat::fltp> r0 = arma::sqrt(1+arma::pow(z/Rout,2));
	arma::Row<rat::fltp> b0 = arma::sqrt(1+arma::pow(z/Rin,2));
	arma::Row<rat::fltp> r1 = arma::sqrt(1+arma::pow((z+height)/Rout,2));
	arma::Row<rat::fltp> b1 = arma::sqrt(1+arma::pow((z+height)/Rin,2));

	// calculate field
	arma::Row<rat::fltp> Bz = -(arma::Datum<rat::fltp>::mu_0*M/2)*(1/r1-1/b1-1/r0+
		1/b0+arma::log((1+r0)%(1+b1)/((1+b0)%(1+r1))));
	
	// return calculate value
	return Bz;
}

// main
int main(){
	// settings
	int num_exp = 8;
	rat::fltp num_refine = 50;
	arma::uword Nt = 1000;

	// test geometry
	rat::fltp Rin = 0.1;
	rat::fltp Rout = 0.12;
	rat::fltp height = 0.02;
	arma::uword nr = 5;
	arma::uword nh = 5;
	arma::uword nl = 180;
	rat::fltp M = 1e6;
	
	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// create armadillo timer
	arma::wall_clock timer;

	// tell user what this thing does
	lg->newl();
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sValidation script checking%s\n",KCYN,KNRM);
	lg->msg("%sthe field on the axis of a%s\n",KCYN,KNRM);
	lg->msg("%saxially magnetised cylinder%s\n",KCYN,KNRM);
	lg->msg("%swith analytical expressions.%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");

	// start setup 
	lg->msg(2,"%s%sGEOMETRY%s\n",KBLD,KGRN,KNRM);
	lg->msg("ring with inner radius %s%.2f [m]%s\n",KYEL,Rin,KNRM);
	lg->msg("outer radius %s%.2f [m]%s and height %s%.2f [m]%s\n",KYEL,Rout,KNRM,KYEL,height,KNRM);
	lg->msg("with magnetisation %s%.2e [MA/m]%s\n",KYEL,M/1e6,KNRM);
	lg->msg(-2,"\n");

	// create magnetised mesh
	rat::fmm::ShMomentSourcesPr mysources = rat::fmm::MomentSources::create();
	mysources->setup_ring_magnet(Rin,Rout,height,nr,nh,nl,0,M);

	// target points
	arma::Mat<rat::fltp> Rt(3,Nt,arma::fill::zeros);
	Rt.row(2) = arma::linspace<arma::Row<rat::fltp> >(-0.2,0.4,Nt);

	// create target point object
	rat::fmm::ShTargetPointsPr mytargets = rat::fmm::MgnTargets::create(Rt);
	mytargets->set_field_type('B',3);

	// setup and run MLFMM
	rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(mysources,mytargets);

	// create settings
	rat::fmm::ShSettingsPr settings = myfmm->settings();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup mlfmm
	myfmm->setup(lg); 

	// run mlfmm
	myfmm->calculate(lg);	// report

	// get results
	arma::Mat<rat::fltp> Bfmm = mytargets->get_field('B');	
	arma::Row<rat::fltp> Bzfmm = Bfmm.row(2);

	// analytical magnetised cylinder
	arma::Row<rat::fltp> Bz = analytic_radially_magnetised_ring_axis(
		Rt.row(2)-height/2, M, Rin, Rout, height);

	// calculate difference between analytic and mlfmm
	arma::Row<rat::fltp> diff = 100*arma::abs(Bfmm.row(2) - Bz)/(arma::max(Bz)-arma::min(Bz));

	lg->msg(2,"%s%sANALYTIC CALCULATION%s\n",KBLD,KGRN,KNRM);
	lg->msg("difference with fmm: %s%2.2f pct%s\n",KYEL,arma::max(diff),KNRM);
	lg->msg(-2,"\n");

	// report result
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// checking if vector potential descending
	if(arma::max(diff)<2){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("difference in magnetic field exceeds tolerance");
	}
	lg->msg(-2,"\n");

	// return
	return 0;
}