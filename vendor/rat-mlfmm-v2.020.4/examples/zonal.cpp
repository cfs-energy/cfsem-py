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
// #include "lptargets.hh"
#include "currentsources.hh"
#include "mlfmm.hh"
#include "currentmesh.hh"
#include "extra.hh"

// main
int main(){
	// settings
	const int num_exp = 7;
	const arma::uword num_refine = 240;

	// geometry
	const rat::fltp Rin = 0.1;
	const rat::fltp Rout = 0.11;
	const rat::fltp height = 0.1;
	const rat::fltp J = 400e6;
	const arma::uword num_turns = 1000;

	// tolerance
	const rat::fltp tol = 2e-2;

	// number of elements for mlfmm
	//rat::fltp dl = 1.2e-3;
	const rat::fltp dl = 1.2e-3;
	const arma::uword num_rad = std::max(2,(int)std::ceil((Rout-Rin)/dl));
	const arma::uword num_height = std::max(2,(int)std::ceil(height/dl));
	const arma::uword num_azym = std::max(2,(int)std::ceil((2*arma::Datum<rat::fltp>::pi*Rin)/dl));

	// number of elements for soleno
	const arma::uword num_layer = 5;

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// combine coords
	arma::Col<rat::fltp> Rt{0,0,0};

	// // targets on line in radial direction
	// rat::fmm::ShLpTargetsPr tar = rat::fmm::LpTargets::create(Rt);
	// tar->set_field_type('R',2*rat::fmm::Extra::polesize(num_exp));
	// tar->add_field_type('I',2*rat::fmm::Extra::polesize(num_exp));

	// // create sources
	// rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create();
	// src->setup_solenoid(Rin,Rout,height,num_rad,num_height,num_azym,J);

	// // create mlfmm
	// rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(src,tar);

	// // create settings
	// rat::fmm::ShSettingsPr settings = myfmm->settings();
	// settings->set_num_exp(num_exp);
	// settings->set_num_refine(num_refine);

	// // setup mlfmm
	// myfmm->setup(lg);

	// // run multipole method
	// myfmm->calculate(lg);

	// // get results
	// arma::Col<rat::fltp> Lp = tar->get_field('R');

	// lg->msg("Localpole");
	// rat::fmm::Extra::display_pole(num_exp,Lp,lg);

	// return
	return 0;
}