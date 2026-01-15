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
#include "tetrahedron.hh"

// main
int main(){
	// settings
	double D = 0.5; // box side length

	// define box corners
	arma::Col<double> R0 = {-D/2,-D/2,-D/2}; 
	arma::Col<double> R1 = {+D/2,-D/2,-D/2}; 
	arma::Col<double> R2 = {+D/2,+D/2,-D/2}; 
	arma::Col<double> R3 = {-D/2,+D/2,-D/2};
	arma::Col<double> R4 = {-D/2,-D/2,+D/2}; 
	arma::Col<double> R5 = {+D/2,-D/2,+D/2}; 
	arma::Col<double> R6 = {+D/2,+D/2,+D/2}; 
	arma::Col<double> R7 = {-D/2,+D/2,+D/2}; 

	// assemble matrix
	arma::Mat<double> Rn(3,8);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;
	Rn.col(4) = R4; Rn.col(5) = R5; Rn.col(6) = R6; Rn.col(7) = R7;

	// get hexahedron to tetrahedron matrix
	arma::Mat<arma::uword>::fixed<5,4> M = Hexahedron::tetrahedron_conversion_matrix();

	// volume of tetrahedron from planes
	arma::Row<double> V = Tetrahedron::calc_volume(Rn,M.t());

	// volume should be 1/6 of the box volume
	assert(arma::all(arma::abs(V.cols(0,3)-std::pow(D,3)/6)<1e-12));
	assert(arma::all(arma::abs(V.col(4)-std::pow(D,3)/3)<1e-12));

	// return
	return 0;
}