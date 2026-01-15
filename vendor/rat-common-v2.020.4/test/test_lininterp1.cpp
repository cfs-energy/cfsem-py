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
// tests linear interpolation function

// main
int main(){
	// settings
	const arma::uword num_interp = 100;
	const rat::fltp xmin = RAT_CONST(0.2);
	const rat::fltp xmax = RAT_CONST(8.4);
	const rat::fltp ximin = RAT_CONST(-0.4);
	const rat::fltp ximax = RAT_CONST(14.2);
	const arma::uword num_vals = 12000;
	const rat::fltp scale = 20;

	// create arrays
	const arma::Col<rat::fltp> x = arma::linspace<arma::Col<rat::fltp> >(xmin,xmax,num_interp);
	const arma::Col<rat::fltp> y = arma::Col<rat::fltp>(num_interp,arma::fill::randu)*scale-scale/2;
	const arma::Col<rat::fltp> xi = arma::Col<rat::fltp>(num_vals,arma::fill::randu)*(ximax-ximin)+ximin;

	// allocate output values 
	arma::Col<rat::fltp> yi_reference, yi_lininterp;

	// perform interpolation
	rat::cmn::Extra::interp1(x,y,xi,yi_reference,"linear",true);
	rat::cmn::Extra::lininterp1f(x,y,xi,yi_lininterp,true);

	// check result
	if(arma::any((yi_lininterp-yi_reference)>1e-5))
		rat_throw_line("linear interpolation does not agree with reference");

	// return
	return 0;
}