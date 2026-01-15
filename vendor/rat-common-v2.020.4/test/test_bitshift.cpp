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
	// generate test numbers
	const arma::uword N = 1000; const rat::fltp max = 1000;
	const arma::Row<arma::uword> vals = 
		arma::conv_to<arma::Row<arma::uword> >::from(
		arma::floor(arma::Row<rat::fltp>(1,N,arma::fill::randu)*max));
	
	// check normal workings
	if(arma::all((vals*2)!=rat::cmn::Extra::bitshift(vals,1)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +1");
	if(arma::all((vals/2)!=rat::cmn::Extra::bitshift(vals,-1)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -1");
	if(arma::all((vals*4)!=rat::cmn::Extra::bitshift(vals,2)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +2");
	if(arma::all((vals/4)!=rat::cmn::Extra::bitshift(vals,-2)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -2");
	if(arma::all((vals*8)!=rat::cmn::Extra::bitshift(vals,3)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +3");
	if(arma::all((vals/8)!=rat::cmn::Extra::bitshift(vals,-3)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -3");
	if(arma::all((vals*16)!=rat::cmn::Extra::bitshift(vals,4)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +4");
	if(arma::all((vals/16)!=rat::cmn::Extra::bitshift(vals,-4)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -4");
	if(arma::all((vals*32)!=rat::cmn::Extra::bitshift(vals,5)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +5");
	if(arma::all((vals/32)!=rat::cmn::Extra::bitshift(vals,-5)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -5");
	if(arma::all((vals*64)!=rat::cmn::Extra::bitshift(vals,6)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for +6");
	if(arma::all((vals/64)!=rat::cmn::Extra::bitshift(vals,-6)))
		rat_throw_line("rat::cmn::Extra::bitshift inconsistent for -6");

	// return
	return 0;
}