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
#include "stars.hh"
#include "mlfmm.hh"

// main
int main(){
	// settings
	const arma::uword Nstars = 10000;
	const rat::fltp fsoft = 0.001;
	const arma::uword num_exp = 7;
	const arma::uword num_refine = 100;

	// create logger
	rat::cmn::ShLogPr lg =rat::cmn::Log::create();	

	lg->newl();
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sValidation script comparing the results%s\n",KCYN,KNRM);
	lg->msg("%sof the mlfmm to that of direct calculation%s\n",KCYN,KNRM);
	lg->msg("%sfor astrodynamics.%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");	

	// set random seed
	arma::arma_rng::set_seed(1002);

	// create source coordinates
	arma::Mat<rat::fltp> Rstars = rat::cmn::Extra::random_coordinates(0, 0, 0, 1.2, Nstars);

	// assign solar masses
	arma::Row<rat::fltp> Mstars = 0.5+arma::Row<rat::fltp>(Nstars,arma::fill::randu);

	// assign softness factors
	arma::Row<rat::fltp> epss = fsoft*arma::Row<rat::fltp>(Nstars,arma::fill::ones);

	// create sources
	rat::fmm::ShStarsPr src = rat::fmm::Stars::create(Rstars,Mstars,epss);
	rat::fmm::ShTargetPointsPr tar = src;
	
	// create mlfmm
	rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create();
	myfmm->set_sources(src); myfmm->set_targets(tar);

	// create settings
	rat::fmm::ShSettingsPr settings = myfmm->settings();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup mlfmm
	myfmm->setup(lg);

	// run direct calculation
	myfmm->calculate_direct(lg);

	// get gravitational potential
	arma::Row<rat::fltp> V1 = tar->get_field('V');
	arma::Mat<rat::fltp> F1 = tar->get_field('F');

	// run multipole method
	myfmm->calculate(lg);

	// get gravitational potential
	arma::Row<rat::fltp> V2 = tar->get_field('V');
	arma::Mat<rat::fltp> F2 = tar->get_field('F');

	// compare two results
	rat::fltp Vdiff = arma::max(arma::abs(V1-V2)/arma::max(arma::abs(V1)));
	rat::fltp Fdiff = arma::max(arma::max(arma::abs(F1-F2)))/arma::max(rat::cmn::Extra::vec_norm(F1));

	// tolerance
	rat::fltp tol = 1e-2;

	// output difference
	lg->msg(2,"%s%sRESULTS%s\n",KBLD,KGRN,KNRM);
	lg->msg("difference in potential is %s%2.4f [pct]%s\n",KYEL,Vdiff*100,KNRM);
	lg->msg("difference in force is %s%2.4f [pct]%s\n",KYEL,Fdiff*100,KNRM);
	lg->msg(-2,"\n");

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);
	
	//std::cout<<arma::join_horiz(F1.t(),F2.t())<<std::endl;
	//std::cout<<arma::join_horiz(Bsol.t(),Bfmm.t())<<std::endl;

	// checking if vector potential descending
	if(Vdiff<tol){
		lg->msg("= accuracy gravitational potential: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy gravitational potential: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("gravitational potential is outside tolerance");
	}
	
	// checking if vector potential descending
	if(Fdiff<tol){
		lg->msg("= accuracy gravitational forces: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy gravitational forces: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("gravitational forces are outside tolerance");
	}

	// go back
	lg->msg(-2,"\n");

	// return
	return 0;
}