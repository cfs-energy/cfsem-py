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
#include "mgncharges.hh"
#include "mgntargets.hh"
#include "mlfmm.hh"

// main
int main(){
	// settings
	const arma::uword Ns = 10000;
	const rat::fltp fsoft = 0.001;
	const arma::uword num_exp = 7;
	const arma::uword num_refine = 100;
	const rat::fltp tol = 1e-2;

	// create logger
	rat::cmn::ShLogPr lg =rat::cmn::Log::create();	

	// set random seed
	arma::arma_rng::set_seed(1002);

	// create source coordinates
	arma::Mat<rat::fltp> Rs = rat::cmn::Extra::random_coordinates(0, 0, 0, 1.2, Ns);

	// assign solar masses
	arma::Row<rat::fltp> sigma_s = -1.0+2.0*arma::Row<rat::fltp>(Ns,arma::fill::randu);

	// assign softness factors
	arma::Row<rat::fltp> epss = fsoft*arma::Row<rat::fltp>(Ns,arma::fill::ones);

	// create sources
	rat::fmm::ShMgnChargesPr src = rat::fmm::MgnCharges::create(Rs,sigma_s,epss);
	rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rs);
	tar->set_field_type("SH",{1,3});

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

	// get scalar potential
	arma::Row<rat::fltp> S1 = tar->get_field('S');
	arma::Mat<rat::fltp> H1 = tar->get_field('H');

	// run multipole method
	myfmm->calculate(lg);

	// get magnetic field
	arma::Row<rat::fltp> S2 = tar->get_field('S');
	arma::Mat<rat::fltp> H2 = tar->get_field('H');

	// compare two results
	rat::fltp Sdiff = arma::max(arma::abs(S1-S2)/arma::max(arma::abs(S1)));
	rat::fltp Hdiff = arma::max(arma::max(arma::abs(H1-H2)))/arma::max(rat::cmn::Extra::vec_norm(H1));
	
	// std::cout<<arma::join_vert(S1,S2,H1,H2).t()<<std::endl;

	// output difference
	lg->msg(2,"%s%sRESULTS%s\n",KBLD,KGRN,KNRM);
	lg->msg("difference in potential is %s%2.4f [pct]%s\n",KYEL,Sdiff*100,KNRM);
	lg->msg("difference in magnetic field is %s%2.4f [pct]%s\n",KYEL,Hdiff*100,KNRM);
	lg->msg(-2,"\n");

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);
	
	// checking if vector potential descending
	if(Sdiff<tol){
		lg->msg("= accuracy scalar potential: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy scalar potential: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("gravitational potential is outside tolerance");
	}
	
	// checking if vector potential descending
	if(Hdiff<tol){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("gravitational forces are outside tolerance");
	}

	// go back
	lg->msg(-2,"\n");

	// return
	return 0;
}