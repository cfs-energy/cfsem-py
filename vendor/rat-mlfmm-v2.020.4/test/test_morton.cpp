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
#include "grid.hh"

// main
int main(){
	// settings
	const arma::uword num_level_min = 2;
	const arma::uword num_level_max = 10;
	const arma::uword num_indexes = 10;

	// set random number generator seed
	arma::arma_rng::set_seed(1001);

	// make list of levels
	const arma::Row<arma::uword> level = 
		arma::regspace<arma::Row<arma::uword> >(
			num_level_min,num_level_max);

	// walk over levels
	for(arma::uword i=0;i<level.n_elem;i++){
		// get level
		const arma::uword mylevel = level(i);

		// calculate number of boxes
		const arma::uword grid_dim = int(std::pow(2,mylevel));
		const arma::uword num_boxes = grid_dim*grid_dim*grid_dim;

		// generate list of random morton indices
		const arma::Row<arma::uword> morton1 = 
			arma::conv_to<arma::Row<arma::uword> >::from(
			arma::Row<rat::fltp>(num_indexes,arma::fill::randu)*rat::fltp(num_boxes-1));

		// convert morton index to grid index
		const arma::Mat<arma::uword> grid_index = rat::fmm::Grid::morton2grid(morton1);

		// check that the index remained the same
		if(!arma::all(arma::all(grid_index>=0)))rat_throw_line(
			"morton2grid created a grid index below zero");
		if(!arma::all(arma::all(grid_index<grid_dim)))rat_throw_line(
			"morton2grid created a grid index larger than the grid");

		// convert back to morton index
		const arma::Row<arma::uword> morton2 = 
			rat::fmm::Grid::grid2morton(grid_index, mylevel);

		// check that the index remained the same
		if(!arma::all(morton1==morton2))rat_throw_line(
			"morton indexing is inconsistent");
	}

	// return
	return 0;
}