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

// DESCRIPTION
// tests cross product and dot product

// main
int main(){
	// settings
	const arma::uword num_vec = 10000;
	
	// set random seed
	arma::arma_rng::set_seed(1001);

	// generate random vector
	arma::Mat<rat::fltp> R = arma::randu(3,num_vec) - RAT_CONST(0.5);

	// in case the universe is about to collapse
	R.cols(arma::find(rat::cmn::Extra::vec_norm(R)==RAT_CONST(0.0))).fill(1.0);

	// normalize
	const arma::Mat<rat::fltp> V = rat::cmn::Extra::normalize(R);

	// check
	if(arma::any(arma::abs(rat::cmn::Extra::vec_norm(V)-RAT_CONST(1.0))>1e-12))
		rat_throw_line("all vector lengths must be equal to one");

	// return
	return 0;
}