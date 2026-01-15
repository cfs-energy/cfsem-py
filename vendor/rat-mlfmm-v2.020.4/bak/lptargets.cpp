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

// include header file
#include "lptargets.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	LpTargets::LpTargets(){

	}

	// constructor
	LpTargets::LpTargets(const arma::Mat<fltp> &Rt){
		set_target_coords(Rt);
	}
	
	// factory
	ShLpTargetsPr LpTargets::create(){
		return std::make_shared<LpTargets>();
	}
	
	// factory
	ShLpTargetsPr LpTargets::create(const arma::Mat<fltp> &Rt){
		return std::make_shared<LpTargets>(Rt);
	}

	// setup localpole to target
	void LpTargets::setup_localpole_to_target(const arma::Mat<fltp> &dR, const ShSettingsPr &/*stngs*/){
		dR_ = dR;
	}
	
	// localpole to target
	void LpTargets::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target, 
		const arma::uword /*num_dim*/, 
		const ShSettingsPr &stngs){

		// get number of expansions
		const arma::uword num_exp = stngs->get_num_exp();

		// allocate output localpoles
		arma::Mat<std::complex<fltp> > Lpt(Extra::polesize(num_exp),dR_.n_cols);

		// walk over localpoles
		for(arma::uword i=0;i<first_target.n_elem;i++){
			// number of targets
			const arma::uword num_targets = last_target(i) - first_target(i) + 1;

			// create transformation matrix
			const ShFmmMat_Lp2LpPr lp2lp = FmmMat_Lp2Lp::create(num_exp, dR_.cols(first_target(i),last_target(i)));

			// calculate localpole at target position
			for(arma::uword k=0;k<num_targets;k++)
				Lpt.col(first_target(i) + k) = lp2lp->apply(k,Lp);
		}

		// set field
		if(has('R'))add_field('R', arma::real(Lpt), false);
		if(has('I'))add_field('I', arma::imag(Lpt), false);
	}

}}