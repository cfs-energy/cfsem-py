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
#include "momentsources.hh"

// default constructor
MomentSources::MomentSources(){
	// set number of dimensions
	set_src_num_dim(3);
}

// factory
ShMomentSourcesPr MomentSources::create(){
	//return ShMomentSourcesPr(new MomentSources);
	return std::make_shared<MomentSources>();
}

// set effective current density
void MomentSources::set_moments(
	const arma::Mat<double> &Meff){
	
	// set effective current
	Meff_ = Meff;
}

// calculation of vector potential for all sources
void MomentSources::calc_direct(ShTargetsPr &tar) const{
	// check moments
	assert(Meff_.n_rows==src_num_dim_);

	// get target coordinates
	arma::Mat<double> Rt = tar->get_target_coords();

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", Savart::calc_M2A(
			Rs_, Meff_, Rt, true), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", Savart::calc_M2H(
			Rs_, Meff_, Rt, true), true);
	}

}

// calculation of vector potential for specific sources
void MomentSources::calc_direct(
	ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{

	// get target coordinates
	arma::Mat<double> Rt = tar->get_target_coords(tidx);

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", tidx, Savart::calc_M2A(
			Rs_.cols(sidx), Meff_.cols(sidx), Rt, false), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", tidx, Savart::calc_M2H(
			Rs_.cols(sidx), Meff_.cols(sidx), Rt, false), true);
	}
}

// setup source to multipole matrices
void MomentSources::setup_source_to_multipole(
	const arma::Mat<double> &dR,
	arma::uword num_exp){
	
	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		// set relative position of the elements 
		// with respect to their multipole
		dR_ = dR;
	}

	// maximize speed over memory efficiency
	// this sets up the full source to multipole matrix in advance
	// this saves some computation time and could be viable when 
	// a relatively small system needs to be calculated many times
	else{
		// set number of expansions and setup matrix for all elements
		M_M_.set_num_exp(num_exp);
		M_M_.calc_matrix(-dR);
	}
}

// get multipole contribution of the sources with indices
// the contributions of the sources are already summed
arma::Mat<std::complex<double> > MomentSources::source_to_multipole(
	const arma::Row<arma::uword> &indices,
	arma::uword num_exp) const{

	// check moments
	assert(Meff_.n_rows==src_num_dim_);

	// allocate output multipole
	arma::Mat<std::complex<double> > Mp;

	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		// check if dR was set
		assert(!dR_.is_empty());

		// setup source to multipole matrix specifically for these elememts
		StMat_So2Mp_M M_M;
		M_M.set_num_exp(num_exp);
		M_M.set_use_parallel(false);
		M_M.calc_matrix(-dR_.cols(indices));
		
		// calculate contribution of these sources to the multipole
		Mp = M_M.apply(Meff_.cols(indices));
	}

	// faster less memory efficient implementation
	else{
		// check if matrix was set
		assert(!M_M_.is_empty());

		// calculate contribution of these sources to the multipole
		Mp = M_M_.apply(Meff_.cols(indices),indices);
	}
	
	// return multipole
	return Mp;
}

// display internal storage
void MomentSources::display() const{
	std::printf("Current sources with");
}


