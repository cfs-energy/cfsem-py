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
#include "currentsources.hh"

// default constructor
CurrentSources::CurrentSources(){
	// set number of dimensions
	set_src_num_dim(3);
}

// factory
ShCurrentSourcesPr CurrentSources::create(){
	//return ShCurrentSourcesPr(new CurrentSources);
	return std::make_shared<CurrentSources>();
}

// set sources from coordinates and currents
void CurrentSources::set_points(
	const arma::Mat<double> &R, 
	const arma::Mat<double> &Ieff){
	set_coords(R); set_currents(Ieff);
}

// set sources by combining other source objects
void CurrentSources::set_points(
	arma::field<ShCurrentSourcesPr> &srcs){

	// get number of meshes
	arma::uword num_srcs = srcs.n_elem;

	// downcast mesh array
	arma::field<ShPointSourcesPr> psrcs(num_srcs);
	for(arma::uword i=0;i<num_srcs;i++)
		psrcs(i) = std::dynamic_pointer_cast<PointSources>(srcs(i));

	// forward construction to mesh class
	PointSources::set_points(psrcs);

	// allocate information arrays
	arma::Row<arma::uword> num_points_each(num_srcs);

	// acquire data from each mesh
	for(arma::uword i=0;i<num_srcs;i++){
		// get numberof nodes
		num_points_each(i) = srcs(i)->num_sources();
	}

	// create node indexing array
	arma::Row<arma::uword> idx(num_srcs+1); idx(0) = 0;
	idx.cols(1,num_srcs) = arma::cumsum(num_points_each,1);

	// allocate current
	Ieff_.set_size(3,num_sources_);

	// get currents
	for(arma::uword i=0;i<num_srcs;i++)
		Ieff_.cols(idx(i),idx(i+1)-1) = srcs(i)->get_currents();

}

// set effective current density
void CurrentSources::set_currents(
	const arma::Mat<double> &Ieff){
	// set effective current
	Ieff_ = Ieff;
}


// get effective current density
arma::Mat<double> CurrentSources::get_currents() const{
	return Ieff_;
}

// calculation of vector potential for all sources
void CurrentSources::calc_direct(ShTargetsPr &tar) const{
	// get target coordinates
	const arma::Mat<double> Rt = tar->get_target_coords();

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", Savart::calc_I2A(
			Rs_, Ieff_, Rt, true), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", Savart::calc_I2H(
			Rs_, Ieff_, Rt, true), true);
	}

}

// calculation of vector potential for specific sources
void CurrentSources::calc_direct(
	ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{

	// check currents
	assert(Ieff_.n_rows==src_num_dim_);

	// get target coordinates
	const arma::Mat<double> Rt = tar->get_target_coords(tidx);

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", tidx, Savart::calc_I2A(
			Rs_.cols(sidx), Ieff_.cols(sidx), Rt, false), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", tidx, Savart::calc_I2H(
			Rs_.cols(sidx), Ieff_.cols(sidx), Rt, false), true);
	}
}

// setup source to multipole matrices
void CurrentSources::setup_source_to_multipole(
	const arma::Mat<double> &dR, 
	const arma::uword num_exp){
	
	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		dR_ = dR;
	}

	// maximize speed over memory efficiency
	else{
		// set number of expansions and setup matrix
		M_J_.set_num_exp(num_exp);
		M_J_.calc_matrix(-dR);
	}	
}

// get multipole contribution of the sources with indices
// the contributions of the sources are already summed
arma::Mat<std::complex<double> > CurrentSources::source_to_multipole(
	const arma::Row<arma::uword> &indices, const arma::uword num_exp) const{
	
	// check currents
	assert(Ieff_.n_rows==3);

	// allocate output multipole
	arma::Mat<std::complex<double> > Mp;

	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		// calculate contribution of currents and return calculated multipole
		StMat_So2Mp_J M_J;
		M_J.set_num_exp(num_exp);
		M_J.calc_matrix(-dR_.cols(indices));
		Mp = M_J.apply(Ieff_.cols(indices));
	}

	// faster less memory efficient implementation
	else{
		Mp = M_J_.apply(Ieff_.cols(indices),indices);
	}

	// return multipole
	return Mp;
}

