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
#include "error.hh"
#include "extra.hh"

// tests cross product and dot product

// main
int main(){
	// settings
	const arma::uword num_vec = 1000;
	const arma::uword max_val = 40;

	// set random seed
	arma::arma_rng::set_seed(1001);

	// make array with random integers
	arma::Row<arma::uword> myints = 
		arma::conv_to<arma::Row<arma::uword> >::from(
		arma::round(arma::Row<rat::fltp>(
		num_vec,arma::fill::randu)*max_val));

	// sort array
	const arma::Row<arma::uword> mysortedints = arma::sort(myints);

	// run find sections
	const arma::Mat<arma::uword> sections = rat::cmn::Extra::find_sections(mysortedints);

	// get indexes
	const arma::Row<arma::uword> first_idx = sections.row(0);
	const arma::Row<arma::uword> last_idx = sections.row(1);

	// get first and last values
	const arma::Row<arma::uword> first_value = mysortedints.cols(first_idx);
	const arma::Row<arma::uword> last_value = mysortedints.cols(last_idx);

	// check that they are the same
	assert(arma::all(first_value==last_value));	
	assert(arma::all(first_idx.tail_cols(first_idx.n_cols-1)
		==last_idx.head_cols(last_idx.n_cols-1)+1));

	// walk over sections
	for(arma::uword i=0;i<sections.n_cols;i++){
		// check if all values are the same
		if(!arma::all(mysortedints.cols(first_idx(i),last_idx(i))==first_value(i)))
			rat_throw_line("find sections is inconsistent");
	}

	// return
	return 0;
}