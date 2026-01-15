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

#include "hexahedron.hh"
#include "integration.hh"

// main
int main(){
	// target point
	arma::Col<double> Rt = {12e-3,15e-3,1e-3};
	// arma::Col<double> Rt = {1e-3,0,0};
	arma::sword num_gauss = 9;
	
	// settings
	double D = 0.01; // box side length
	double dxtrap = 0.001;

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

	// setup mesh
	arma::Col<arma::uword> nh = {0,1,2,3,4,5,6,7};

	// self field or not?	
	arma::Row<arma::uword> ch = Hexahedron::is_inside(Rt,Rn,nh);
	bool is_self = ch(0)==1;

	// volume
	double Ve = arma::as_scalar(Hexahedron::calc_volume(Rn,nh));

	// calculate
	std::cout<<is_self<<std::endl;
	arma::Col<double> H1 = Integration::calc_analytic(Rt,Rn);
	arma::Col<double> H2 = Integration::calc_numeric_volume(Rt,Rn,Ve,num_gauss);
	arma::Col<double> H3 = Integration::calc_numeric_surface(Rt,Rn,num_gauss,is_self);
	arma::Col<double> H4 = Integration::calc_savart(Rt,Rn,Ve,is_self);
	arma::Col<double> H5 = Integration::calc_analytic_tet(Rt,Rn);

	// display result
	std::cout<<arma::join_horiz(arma::join_horiz(arma::join_horiz(arma::join_horiz(H1,H2),H3),H4),H5)<<std::endl;

	// return
	return 0;
}