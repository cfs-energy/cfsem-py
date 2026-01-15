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

#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "hexahedron.hh"

// main
int main(){
	// settings
	double D = 0.8; // box side length
	arma::uword N = 100;
	double dxtrap = 0.05;
	double a = 0.4;
	double b = 0.7;
	double c = 0.6;

	// define box corners
	arma::Col<double> R0 = {-D/2-dxtrap,-D/2,-D/2}; 
	arma::Col<double> R1 = {+D/2+dxtrap,-D/2,-D/2}; 
	arma::Col<double> R2 = {+D/2,+D/2,-D/2}; 
	arma::Col<double> R3 = {-D/2,+D/2,-D/2};
	arma::Col<double> R4 = {-D/2-dxtrap,-D/2,+D/2}; 
	arma::Col<double> R5 = {+D/2+dxtrap,-D/2,+D/2}; 
	arma::Col<double> R6 = {+D/2,+D/2,+D/2}; 
	arma::Col<double> R7 = {-D/2,+D/2,+D/2}; 

	// assemble matrix with nodes
	arma::Mat<double> Rn(3,8);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;
	Rn.col(4) = R4; Rn.col(5) = R5; Rn.col(6) = R6; Rn.col(7) = R7;

	// set diagonal gradient
	arma::Row<double> phi = a*Rn.row(0) + b*Rn.row(1) + c*Rn.row(2);

	// derivative
	arma::Mat<double> Rqtest(3,N,arma::fill::randu);
	arma::Mat<double> dphi = Hexahedron::quad2cart_derivative(Rn,Rqtest,phi);

	// check output
	assert(arma::all(arma::abs(dphi.row(0)-a)<1e-6));
	assert(arma::all(arma::abs(dphi.row(1)-b)<1e-6));
	assert(arma::all(arma::abs(dphi.row(2)-c)<1e-6));
}
