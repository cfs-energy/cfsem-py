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
#include "pointsources.hh"


// constructor
PointSources::PointSources(){

}

// set points by combining other point sources
void PointSources::set_points(
	arma::field<ShPointSourcesPr> &srcs){
	// get number of meshes
	arma::uword num_srcs = srcs.n_elem;

	// allocate information arrays
	arma::Row<arma::uword> num_points_each(num_srcs);

	// acquire data from each mesh
	for(arma::uword i=0;i<num_srcs;i++){
		// get numberof nodes
		num_points_each(i) = srcs(i)->num_sources();
	}

	// set number of sources
	num_sources_ = arma::sum(num_points_each);

	// create node indexing array
	arma::Row<arma::uword> idx(num_srcs+1); idx(0) = 0;
	idx.cols(1,num_srcs) = arma::cumsum(num_points_each,1);

	// allocate current
	Rs_.set_size(3,num_sources_);

	// get currents
	for(arma::uword i=0;i<num_srcs;i++)
		Rs_.cols(idx(i),idx(i+1)-1) = srcs(i)->get_source_coords();

}

// count number of sources stored
arma::uword PointSources::num_sources() const{
	// return number of elements
	return num_sources_;
}

// set element coordinates without currents
void PointSources::set_coords(
	const arma::Mat<double> &Rs){
	
	// check user input
	assert(Rs.n_rows==3);
	assert(Rs.is_finite());

	// set internal coordinates 
	Rs_ = Rs;

	// set number of sources
	num_sources_ = Rs_.n_cols;
}

// method for getting all coordinates
arma::Mat<double> PointSources::get_source_coords() const{
	// return coordinates
	return Rs_;
}

// method for getting coordinates with specific indices
arma::Mat<double> PointSources::get_source_coords(
	const arma::Mat<arma::uword> &indices) const{

	// return coordinates
	return Rs_.cols(indices);
}







