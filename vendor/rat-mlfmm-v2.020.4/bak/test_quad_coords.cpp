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

#include "quadrilateral.hh"
#include "common/gauss.hh"

// main
int main(){
	// settings
	double D = 0.8; // box side length
	double tol = 1e-5;
	arma::uword Np = 100;
	double dxtrap = 0.2;

	// define face corners
	arma::Col<double> R0 = {-D/2-dxtrap,-D/2,0}; 
	arma::Col<double> R1 = {+D/2+dxtrap,-D/2,0}; 
	arma::Col<double> R2 = {+D/2,+D/2,0}; 
	arma::Col<double> R3 = {-D/2,+D/2,0};

	// assemble matrix with nodes
	arma::Mat<double> Rn(3,4);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;

	// slope plane
	double a = 0.1; double b = 0.2;
	Rn.row(2) = a*Rn.row(0) + b*Rn.row(1);
	// Rn.swap_rows(2,1);

	// calculate face normal
	arma::Col<double>::fixed<3> N = arma::cross(Rn.col(1)-Rn.col(0), Rn.col(3)-Rn.col(0));
	N = N/arma::as_scalar(Extra::vec_norm(N));

	// make random quadrilateral coordinates
	arma::Mat<double> Rq1 = 4*(arma::Mat<double>(2,Np,arma::fill::randu)-0.5);

	// convert to carthesian coordinates
	arma::Mat<double> Rc = Quadrilateral::quad2cart(Rn,Rq1);

	// offset points by facenormal
	// this to check if they actually go back
	// to the face 
	arma::Row<double> Noff(Np,arma::fill::randu); Noff -= 0.5;
	for(arma::uword i=0;i<3;i++)Rc.row(i) += N(i);

	// and back
	arma::Mat<double> Rq2 = Quadrilateral::cart2quad(Rn,Rc,tol);

	// check if the points remained the same
	assert(arma::all(arma::all(arma::abs(Rq1-Rq2)/(std::sqrt(arma::as_scalar(arma::sum(
	 	(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0))))<0.001)));

	// test mesh 
	// calculate gauss points
	// arma::Mat<double> v = Gauss::calc_gauss_points(5);
	// arma::Row<double> xg = v.row(0); 
	// arma::Row<double> wg = v.row(1);
	// std::cout<<arma::join_horiz(xg.t(),wg.t())<<std::endl;
	// arma::Col<double> Rctest = {0.4,0.4,0.1};
	// arma::Mat<double> Rqtest = Quadrilateral::cart2quad(Rn,Rctest,tol);
	// Rqtest = arma::clamp(Rqtest,-1.0,1.0);
	// std::cout<<Quadrilateral::quad2cart(Rn,Rqtest).t()<<std::endl;
	// arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
	// Quadrilateral::setup_source_grid(Rqgrd, wgrd, Rqtest,xg,wg);
	// arma::Mat<double> Rpoints = Quadrilateral::quad2cart(Rn,Rqgrd);
	// std::cout<<arma::join_horiz(Rpoints.t(),wgrd.t())<<std::endl;
}
	
