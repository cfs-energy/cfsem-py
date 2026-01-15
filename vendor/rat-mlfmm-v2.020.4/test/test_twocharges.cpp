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
	const arma::uword num_exp = 7;
	const arma::uword num_refine = 100;
	const rat::fltp tol = 1e-2;

	// create logger
	rat::cmn::ShLogPr lg =rat::cmn::Log::create();	

	// create target coordinates
	arma::Mat<rat::fltp>::fixed<3,1> Rt{0,0,0};

	// sources
	// negative charge at z = -0.01
	// positive charge at z = 0.01

	// create source coordinates
	arma::Mat<rat::fltp>::fixed<3,2> Rs{0,0,-0.01, 0,0,0.01};

	// assign solar masses
	arma::Row<rat::fltp> sigma_s{-1.0,1.0};

	// assign softness factors
	arma::Row<rat::fltp> epss{0,0};

	// create sources
	rat::fmm::ShMgnChargesPr src = rat::fmm::MgnCharges::create(Rs,sigma_s,epss);
	rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt);
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

	// check if the field is negative
	if(H1(2)>0)rat_throw_line("Expecting negative H-field in the z-direction for Direct calculation");
	if(H2(2)>0)rat_throw_line("Expecting negative H-field in the z-direction for MLFMM calculation");

	// return
	return 0;
}