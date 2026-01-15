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
	const rat::fltp D = RAT_CONST(0.8); // box side length
	const rat::fltp tol = RAT_CONST(1e-6);
	const arma::uword N = 100;
	const rat::fltp dxtrap = RAT_CONST(0.1);
	const arma::sword num_gauss = 4;

	// define box corners
	const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2-dxtrap,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2+dxtrap,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {+D/2,+D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R3 = {-D/2,+D/2,-D/2};
	const arma::Col<rat::fltp>::fixed<3> R4 = {-D/2-dxtrap,-D/2,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R5 = {+D/2+dxtrap,-D/2,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R6 = {+D/2,+D/2,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R7 = {-D/2,+D/2,+D/2};

	// assemble matrix with nodes
	arma::Mat<rat::fltp> Rn(3,8);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;
	Rn.col(4) = R4; Rn.col(5) = R5; Rn.col(6) = R6; Rn.col(7) = R7;

	// setup mesh
	const arma::Col<arma::uword> n{0,1,2,3,4,5,6,7};

	// create gauss points
	const arma::Mat<rat::fltp> Rq = rat::cmn::Hexahedron::create_gauss_points(num_gauss);
	const arma::Mat<rat::fltp> dN = rat::cmn::Hexahedron::shape_function_derivative(Rq.rows(0,2));
	const arma::Mat<rat::fltp> J = rat::cmn::Hexahedron::shape_function_jacobian(Rn,dN);
	const arma::Row<rat::fltp> Jdet = rat::cmn::Hexahedron::jacobian2determinant(J);

	// calculate volume
	const rat::fltp volume1 = arma::accu(Rq.row(3)%Jdet);
	const rat::fltp volume2 = arma::as_scalar(rat::cmn::Hexahedron::calc_volume(Rn,n));

	// check result
	if(std::abs(volume1 - volume2)/std::abs(volume2)>tol)
		rat_throw_line("calculated volume is out of tolerance");

	// return
	return 0;
}