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
// tests the bitshift method added in Extra

// main
int main(){
	// settings
	arma::Row<arma::uword> A{1,2,5,6,8,10,15,20,21,25};
	arma::Row<arma::uword> B{0,3,5,9,12,15,21,33};
	arma::Row<arma::uword> expected{0,0,1,0,0,0,1,0,1,0};

	// run is member function
	arma::Row<arma::uword> is_member = rat::cmn::Extra::ismember(A,B).t();

	// check output
	if(!arma::all(expected==is_member))
		rat_throw_line("output is not as expected");

	// return
	return 0;
}