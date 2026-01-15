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
#include "rat/common/log.hh"
#include "mlfmm.hh"
#include "currentsources.hh"
#include "mgntargets.hh"
#include "multisources.hh"


// main
int main(){
	// settings
	const arma::uword Ns = arma::uword(1e6); // number of current sources
	const arma::uword Nt = arma::uword(1e6); // number of target points
	const arma::uword num_exp = 5; // minimal number of expansions tested
	const rat::fltp current = 1000; // element current in [A]
	const bool mem_efficient = false;
	const bool large_ilist = false;
	const bool van_lanen = true; 
	const arma::uword num_refine = 100;
	const bool enable_gpu = false;

	// set random seed
	arma::arma_rng::set_seed(1002);

	// create quasi random current source coordinates
	const arma::Mat<rat::fltp> Rs = rat::cmn::Extra::random_coordinates(0, 0.1, 0.1, 1, Ns); // x,y,z,size,N
	const arma::Mat<rat::fltp> dRs = rat::cmn::Extra::random_coordinates(0, 0, 0, 2e-3, Ns); // x,y,z,size,N
	const arma::Row<rat::fltp> Is = arma::Row<rat::fltp>(Ns,arma::fill::ones)*current;
	const arma::Row<rat::fltp> epss = 0.7*rat::cmn::Extra::vec_norm(dRs);

	// create target coordinates
	const arma::Mat<rat::fltp> Rt = rat::cmn::Extra::random_coordinates(0, 0, 0, 1, Nt); // x,y,z,size,N
	
	// create fmm
	rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create();

	// set settimgs
	rat::fmm::ShSettingsPr stngs = myfmm->settings();
	stngs->set_large_ilist(large_ilist);
	stngs->set_enable_fmm(true);
	stngs->set_enable_s2t(true);
	stngs->set_num_exp(num_exp);
	stngs->set_num_refine(num_refine);
	stngs->set_memory_efficient_l2t(mem_efficient);
	stngs->set_memory_efficient_s2m(mem_efficient);
	stngs->set_enable_gpu(enable_gpu);

	// create targets
	rat::fmm::ShMgnTargetsPr targets = rat::fmm::MgnTargets::create(Rt);

	// create sources
	rat::fmm::ShCurrentSourcesPr sources = rat::fmm::CurrentSources::create(Rs,dRs,Is,epss);
	sources->set_van_Lanen(van_lanen);

	// add sources and targets to mlfmm
	myfmm->set_sources(sources);
	myfmm->set_targets(targets);

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// setup mlfmm
	myfmm->setup(lg);

	// run multipole method
	myfmm->calculate(lg);

	// report 
	myfmm->display(lg);

	// end
	return 0;
}