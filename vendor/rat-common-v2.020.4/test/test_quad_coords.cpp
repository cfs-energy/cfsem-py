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
#include "gauss.hh"
#include "extra.hh"

// main
int main(){
	// settings
	const rat::fltp D = RAT_CONST(0.8); // box side length
	const rat::fltp tol = RAT_CONST(1e-5);
	const arma::uword Np = 100;
	const rat::fltp dxtrap = RAT_CONST(0.2);

	// define face corners
	const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2-dxtrap,-D/2,0}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2+dxtrap,-D/2,0}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {+D/2,+D/2,0}; 
	const arma::Col<rat::fltp>::fixed<3> R3 = {-D/2,+D/2,0};

	// assemble matrix with nodes
	arma::Mat<rat::fltp> Rn(3,4);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;

	// slope plane
	const rat::fltp a = 0.1; const rat::fltp b = 0.2;
	Rn.row(2) = a*Rn.row(0) + b*Rn.row(1);
	// Rn.swap_rows(2,1);

	// calculate face normal
	arma::Col<rat::fltp>::fixed<3> N = arma::cross(Rn.col(1)-Rn.col(0), Rn.col(3)-Rn.col(0));
	N = N/arma::as_scalar(rat::cmn::Extra::vec_norm(N));

	// make random quadrilateral coordinates
	const arma::Mat<rat::fltp> Rq1 = 4*(arma::Mat<rat::fltp>(2,Np,arma::fill::randu)-0.5);

	// convert to carthesian coordinates
	arma::Mat<rat::fltp> Rc = rat::cmn::Quadrilateral::quad2cart(Rn,Rq1);

	// offset points by facenormal
	// this to check if they actually go back
	// to the face 
	arma::Row<rat::fltp> Noff(Np,arma::fill::randu); Noff -= 0.5;
	for(arma::uword i=0;i<3;i++)Rc.row(i) += Noff*N(i);

	// and back
	const arma::Mat<rat::fltp> Rq2 = rat::cmn::Quadrilateral::cart2quad(Rn,Rc,tol);

	// check if the points remained the same
	// assert(arma::all(arma::all(arma::abs(Rq1-Rq2)/(std::sqrt(arma::as_scalar(arma::sum(
	//  	(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0))))<0.001)));
	if(!arma::all(arma::all(arma::abs(Rq1-Rq2)/(std::sqrt(arma::as_scalar(arma::sum(
	 	(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0))))<0.001))){
		rat_throw_line("quadrilateral coordinate transformation is not consistent");
	}

	// test mesh 
	// calculate gauss points
	// arma::Mat<rat::fltp> v = Gauss::calc_gauss_points(5);
	// arma::Row<rat::fltp> xg = v.row(0); 
	// arma::Row<rat::fltp> wg = v.row(1);
	// std::cout<<arma::join_horiz(xg.t(),wg.t())<<std::endl;
	// arma::Col<rat::fltp> Rctest = {0.4,0.4,0.1};
	// arma::Mat<rat::fltp> Rqtest = rat::cmn::Quadrilateral::cart2quad(Rn,Rctest,tol);
	// Rqtest = arma::clamp(Rqtest,-1.0,1.0);
	// std::cout<<rat::cmn::Quadrilateral::quad2cart(Rn,Rqtest).t()<<std::endl;
	// arma::Mat<rat::fltp> Rqgrd; arma::Row<rat::fltp> wgrd;
	// rat::cmn::Quadrilateral::setup_source_grid(Rqgrd, wgrd, Rqtest,xg,wg);
	// arma::Mat<rat::fltp> Rpoints = rat::cmn::Quadrilateral::quad2cart(Rn,Rqgrd);
	// std::cout<<arma::join_horiz(Rpoints.t(),wgrd.t())<<std::endl;
}
	
