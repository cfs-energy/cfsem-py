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
#include <cassert>

// specific headers
#include "error.hh"
#include "extra.hh"

// main
int main(){
	// settings
	arma::uword list_length = arma::uword(RAT_CONST(1e7));
	rat::fltp max_index = arma::uword(RAT_CONST(1e5));

	// create a list of random numbers
	arma::Row<arma::uword> list = arma::conv_to<arma::Row<arma::uword> >::from(
		arma::Row<rat::fltp>(list_length,arma::fill::randn)*max_index);

	// sort
	arma::Row<arma::uword> list_sorted = rat::cmn::Extra::parsort(list);

	// check sorted
	if(!list_sorted.is_sorted())rat_throw_line("sorting failed");


}