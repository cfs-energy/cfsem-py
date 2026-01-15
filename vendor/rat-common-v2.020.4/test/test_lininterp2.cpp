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
#include "extra.hh"

// input matrix
//     [1,4,7]
// v = [2,5,8]
//     [3,6,9]

// main
int main(){
	// create a matrix
	const arma::Mat<rat::fltp>::fixed<3,3> v{RAT_CONST(1.0),RAT_CONST(2.0),RAT_CONST(3.0),RAT_CONST(4.0),RAT_CONST(5.0),RAT_CONST(6.0),RAT_CONST(7.0),RAT_CONST(8.0),RAT_CONST(9.0)};
	const arma::Col<rat::fltp>::fixed<3> x{RAT_CONST(1.0),RAT_CONST(2.0),RAT_CONST(3.0)};
	const arma::Col<rat::fltp>::fixed<3> y{RAT_CONST(4.0),RAT_CONST(5.0),RAT_CONST(6.0)};

	// location for interpolation
	const arma::Col<rat::fltp>::fixed<4> xi{RAT_CONST(1.5),RAT_CONST(2.5),RAT_CONST(1.5),RAT_CONST(2.5)};
	const arma::Col<rat::fltp>::fixed<4> yi{RAT_CONST(4.5),RAT_CONST(4.5),RAT_CONST(5.5),RAT_CONST(5.5)};

	// allocate output
	arma::Col<rat::fltp> vi;

	// perform interpolation
	rat::cmn::Extra::lininterp2f(x,y,v,xi,yi,vi,false);

	// check output
	if(std::abs(vi(0) - arma::accu(v.submat(0,0,1,1))/4)>RAT_CONST(1e-9))rat_throw_line("first value is not interpolated correctly");
	if(std::abs(vi(1) - arma::accu(v.submat(0,1,1,2))/4)>RAT_CONST(1e-9))rat_throw_line("second value is not interpolated correctly");
	if(std::abs(vi(2) - arma::accu(v.submat(1,0,2,1))/4)>RAT_CONST(1e-9))rat_throw_line("third value is not interpolated correctly");
	if(std::abs(vi(3) - arma::accu(v.submat(1,1,2,2))/4)>RAT_CONST(1e-9))rat_throw_line("fourth value is not interpolated correctly");

	// return
	return 0;
}