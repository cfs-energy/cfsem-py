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
	// create list of values
	const rat::fltp tol = RAT_CONST(0.01);
	arma::Row<rat::fltp> vv = {-2.2,3.1,-1.8,2.1,4.6,0.5,2.1,3.1,3.2,4.4,-2.2};

	// create noisy version of the dataset
	arma::Row<rat::fltp> vv_noisy = vv + (arma::randu()-0.5)*0.99*tol;

	// calculate unique id's within tolerance
	arma::Col<arma::uword> id1 = rat::cmn::Extra::find_unique(vv_noisy,tol);

	// calculate control ids using armadillo unique function on regular data
	arma::Col<arma::uword> id2 = arma::find_unique(vv);

	// check if id lists are identical
	if(arma::any(id1!=id2))rat_throw_line("id lists are not in agreement");

	// return
	return 0;
}