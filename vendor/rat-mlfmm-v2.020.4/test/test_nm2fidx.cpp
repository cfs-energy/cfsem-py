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
#include "extra.hh"

// main
int main(){
	// settings
	const int num_exp = 12;

	// run for single pole
	// allocate multipole array with zeros
	arma::Col<arma::uword> Y1(rat::fmm::Extra::polesize(num_exp),arma::fill::zeros);

	// walk over harmonic content
	// and try to fill multipole with ones
	for(int n=0;n<=num_exp;n++){
		for(int m=-n;m<=n;m++){
			Y1(rat::fmm::Extra::nm2fidx(n,m)) = 1;
		}
	}

	// check output 
	if(!arma::all(Y1==1))rat_throw_line(
		"single multipole indexing is inconsistent");

	// run for rat::fltp pole
	// allocate multipole array with zeros
	arma::Col<arma::uword> Y2(rat::fmm::Extra::dbpolesize(num_exp),arma::fill::zeros);

	// walk over harmonic content
	// and try to fill multipole with ones
	for(int n=0;n<=2*num_exp;n++){
		for(int m=-n;m<=n;m++){
			Y2(rat::fmm::Extra::nm2fidx(n,m)) = 1;
		}
	}

	// check output 
	if(!arma::all(Y2==1))rat_throw_line(
		"single multipole indexing is inconsistent");

	// return
	return 0;
}