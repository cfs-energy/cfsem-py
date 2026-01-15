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
	const rat::fltp tol = RAT_CONST(1e-9);
	const arma::uword N = 100;
	const rat::fltp dxtrap = RAT_CONST(0.1);
	const arma::sword num_gauss = 4;

	// example taken from http://members.toast.net/ahines/fem/jacobi.htm
	// define box corners
	const arma::Col<rat::fltp>::fixed<3> R0 = {1.0,0.0,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {1.0,1.0,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {0.0,1.0,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R3 = {0.0,0.0,0.0};
	const arma::Col<rat::fltp>::fixed<3> R4 = {1.0,0.0,0.75}; 
	const arma::Col<rat::fltp>::fixed<3> R5 = {1.0,1.0,0.75}; 
	const arma::Col<rat::fltp>::fixed<3> R6 = {0.0,1.0,0.75}; 
	const arma::Col<rat::fltp>::fixed<3> R7 = {0.0,0.0,0.75};

	// assemble matrix with nodes
	arma::Mat<rat::fltp> Rn(3,8);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;
	Rn.col(4) = R4; Rn.col(5) = R5; Rn.col(6) = R6; Rn.col(7) = R7;

	// reference jacobian matrix for this element 
	const arma::Col<rat::fltp> Jref{0,-0.5,0,0.5,0.0,0.0,0.0,0.0,0.375};

	// determinant for this jacobian
	const rat::fltp Jdetref = RAT_CONST(3.0)/32;

	// calculate jacobian at center point
	const arma::Col<rat::fltp> Rq = rat::cmn::Hexahedron::create_gauss_points(1).rows(0,2);
	const arma::Col<rat::fltp> dN = rat::cmn::Hexahedron::shape_function_derivative(Rq);
	const arma::Col<rat::fltp> J = rat::cmn::Hexahedron::shape_function_jacobian(Rn,dN);

	// check jacobian matrix
	if(arma::any(arma::abs(J-Jref)>tol))
		rat_throw_line("jacobian deviates from expected");

	// calculate determinant
	const rat::fltp Jdet1 = arma::as_scalar(rat::cmn::Hexahedron::jacobian2determinant(J));

	// check determinant
	if(std::abs(Jdet1-Jdetref)>tol)
		rat_throw_line("determinant deviates from expected");

	// calculate determinant
	const rat::fltp Jdet2 = arma::as_scalar(rat::cmn::Hexahedron::jacobian2determinant(rat::cmn::Hexahedron::jacobian_t(J)));

	// check determinant
	if(std::abs(Jdet2-Jdetref)>tol)
		rat_throw_line("determinant deviates from expected");

	// return
	return 0;
}