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
#include <cassert>

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "mgntargets.hh"
#include "momentsources.hh"
#include "mlfmm.hh"

// analytical field on axis of an axially magnetised finite cylinder
// https://www.physicsforums.com/threads/field-bz-inside-axially-magnetized-permanent-ring-magnet.867355/
arma::Row<rat::fltp> analytic_magnetised_ring_axis(
	const arma::Row<rat::fltp> &z, const rat::fltp M, const rat::fltp Rin, 
	const rat::fltp Rout, const rat::fltp height){

	// analytical equation
	arma::Row<rat::fltp> Bz = (arma::Datum<rat::fltp>::mu_0*M/2)*(
		(height+z)/arma::sqrt(Rout*Rout + (height+z)%(height+z)) - z/arma::sqrt(Rout*Rout+z%z) - 
		((height+z)/arma::sqrt(Rin*Rin+(height+z)%(height+z)) - z/arma::sqrt(Rin*Rin+z%z)));

	// return calculate value
	return Bz;
}

// main
int main(){
	// settings
	const arma::uword num_exp = 8;
	const arma::uword num_refine = 50;
	const arma::uword Nt = 1000;

	// test geometry
	const rat::fltp Rin = 0.1;
	const rat::fltp Rout = 0.12;
	const rat::fltp height = 0.02;
	const arma::uword nr = 5;
	const arma::uword nh = 5;
	const arma::uword nl = 180;
	const rat::fltp M = 1e6;
	
	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// create armadillo timer
	arma::wall_clock timer;

	// tell user what this thing does
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
	mysources->setup_ring_magnet(Rin,Rout,height,nr,nh,nl,M,0);

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
	const arma::Mat<rat::fltp> Bfmm = mytargets->get_field('B');	
	const arma::Row<rat::fltp> Bzfmm = Bfmm.row(2);

	// report analytic solution
	lg->msg(2,"%s%sANALYTIC CALCULATION%s\n",KBLD,KGRN,KNRM);

	// analytical magnetised cylinder
	const arma::Row<rat::fltp> Bz = analytic_magnetised_ring_axis(
		Rt.row(2)-height/2, M, Rin, Rout, height);

	// calculate difference between analytic and mlfmm
	const arma::Row<rat::fltp> diff = 100*arma::abs(Bfmm.row(2) - Bz)/(arma::max(Bz)-arma::min(Bz));

	// report difference
	lg->msg("difference with fmm: %s%2.2f pct%s\n",KYEL,arma::max(diff),KNRM);
	lg->msg(-2,"\n");

	// report result
	lg->msg(2,"%s%sRESULTS%s\n",KBLD,KGRN,KNRM);

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
