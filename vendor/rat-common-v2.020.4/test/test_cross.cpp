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

	// generate random vectors
	arma::Mat<rat::fltp> R1(3,num_vec,arma::fill::randu);
	arma::Mat<rat::fltp> R2(3,num_vec,arma::fill::randu);
	const arma::Row<rat::fltp> dp = rat::cmn::Extra::dot(R1,R2)/(rat::cmn::Extra::vec_norm(R1)%rat::cmn::Extra::vec_norm(R2));

	// remove too similar vectors
	const arma::Row<arma::uword> idx = arma::find(dp<RAT_CONST(0.99)).t();
	R1 = R1.cols(idx); R2 = R2.cols(idx);

	// calculate cross product
	const arma::Mat<rat::fltp> R3 = rat::cmn::Extra::cross(R1,R2);

	// check vector orthogonality
	if(!arma::all(rat::cmn::Extra::dot(R1,R3)<1e-6))
		rat_throw_line("cross product inconsistent");
	if(!arma::all(rat::cmn::Extra::dot(R2,R3)<1e-6))
		rat_throw_line("cross product inconsistent");
	
	// return
	return 0;
}