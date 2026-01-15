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
#include "error.hh"
#include "elements.hh"

// main
int main(){
	// settings
	const rat::fltp tol = 1e-4;
	const arma::sword num_gauss = 2; // for higher values the system is not completely solveable

	// divide corner nodes by 2 to create gauss points
	const arma::Mat<rat::fltp>::fixed<3,8> Rn = arma::conv_to<arma::Mat<rat::fltp> >::from(rat::cmn::Hexahedron::get_corner_nodes()).t();
	
	// create gauss points
	const arma::Mat<rat::fltp> gp = rat::cmn::Hexahedron::create_gauss_points(num_gauss);
	const arma::Mat<rat::fltp> Rqg = gp.rows(0,2);
	const arma::Row<rat::fltp> wg = gp.row(3);

	// values at gauss points
	const arma::Row<rat::fltp> Vg1(Rqg.n_cols,arma::fill::randu);

	// extrapolate to nodes
	const arma::Row<rat::fltp> Vn = rat::cmn::Hexahedron::gauss2nodes(Rn,Rqg,wg,Vg1);

	// revert back to gauss points
	const arma::Row<rat::fltp> Vg2 = rat::cmn::Hexahedron::quad2cart(Vn,Rqg);

	// check result
	if(arma::any(arma::abs(Vg1 - Vg2)>tol))rat_throw_line("result outside of tolerance");

	// return
	return 0;
}