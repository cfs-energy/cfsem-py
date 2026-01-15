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

// include header
#include "mtargets.hh"

// constructor
MTargets::MTargets(){
	// field type when used as target
	const arma::Row<arma::uword> num_dim{3,3};	
	set_field_type("AH",num_dim);
}

// get field from internal storage
arma::Mat<double> MTargets::get_field(
	const std::string &type, arma::uword num_dim) const{

	// make sure that type is only one character
	assert(type.length()==1);

	// allocate output
	arma::Mat<double> Mout;

	// magnetic flux density (special case)
	if(type=="B"){
		// check
		assert(has("H")); //assert(has("M"));

		// recursive call
		Mout = arma::datum::mu_0*Targets::get_field("H",num_dim);
			// + Targets::get_field("M",num_dim));
	}

	// normal cases
	else{
		Mout = Targets::get_field(type,num_dim);
	}

	// add field to matrix
	return Mout; 
}


// setup localpole to target matrix
void MTargets::setup_localpole_to_target(
	const arma::Mat<double> &dR, 
	const arma::uword num_exp){

	// check input
	assert(dR.is_finite());

	// memory efficient implementation (default)
	if(enable_memory_efficient_l2t_){
		dRl2t_ = dR;
	}

	// maximize computation speed
	else{
		// matrix for vector potential
		if(has("A")){
			// calculate matrix for all target points
			M_A_.set_num_exp(num_exp);
			M_A_.calc_matrix(-dR);
		}

		// matrix for magnetic field
		if(has("H")){
			// calculate matrix for all target points
			M_H_.set_num_exp(num_exp);
			M_H_.calc_matrix(-dR);
		}
	}
}

// add field contribution of supplied localpole to targets with indices
void MTargets::localpole_to_target(
	const arma::Mat<std::complex<double> > &Lp, 
	const arma::Row<arma::uword> &indices, 
	const arma::uword num_exp){

	// memory efficient implementation (default)
	// note that this method is called many times in parallel
	// do not re-use the class property of M_A and M_H
	if(enable_memory_efficient_l2t_){
		// check if dR was set
		assert(!dRl2t_.is_empty());

		// vector potential
		if(has("A")){
			// calculate matrix for these target points
			StMat_Lp2Ta_A M_A;
			M_A.set_num_exp(num_exp);
			M_A.calc_matrix(-dRl2t_.cols(indices));

			// calculate and add vector potential
			add_field("A", indices, M_A.apply(Lp), false);
		}

		// magnetic field
		if(has("H")){
			// calculate matrix for these target points
			StMat_Lp2Ta_H M_H;
			M_H.set_num_exp(num_exp);
			M_H.set_use_parallel(false);
			M_H.calc_matrix(-dRl2t_.cols(indices));
			
			// calculate and add magnetic field
			add_field("H", indices, M_H.apply(Lp), false);
		}
	}

	// maximize computation speed using pre-calculated matrix
	else{
		// vector potential
		if(has("A")){
			// check if localpole to target matrix was set
			assert(!M_A_.is_empty());

			// calculate and add vector potential
			add_field("A", indices, M_A_.apply(Lp,indices), false);
		}

		// magnetic field
		if(has("H")){
			// check if localpole to target matrix was set
			assert(!M_H_.is_empty());

			// calculate and add magnetic field
			add_field("H", indices, M_H_.apply(Lp,indices), false);
		}
	}
}