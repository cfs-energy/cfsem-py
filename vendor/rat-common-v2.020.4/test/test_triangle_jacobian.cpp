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

	// define box corners
	const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2-dxtrap,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2+dxtrap,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {-D/2,+D/2,-D/2};

	// assemble matrix with nodes
	arma::Mat<rat::fltp> Rn(3,3);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2;

	// setup mesh
	const arma::Col<arma::uword> n{0,1,2};

	// walk over number of gauss points
	for(arma::uword i=1;i<=8;i++){
		// create gauss points
		// const arma::Mat<rat::fltp> Rq = rat::cmn::Triangle::create_gauss_points(num_gauss);
		const arma::Mat<rat::fltp> gp = rat::cmn::Triangle::create_gauss_points_catmull_clark(i);
		const arma::Mat<rat::fltp> Rq = gp.rows(0,1);
		const arma::Row<rat::fltp> wg = gp.row(2);
		if(std::abs(arma::accu(wg)-1.0/2)>1e-7)rat_throw_line("weights do not sum up to 1/2");

		// const arma::Mat<rat::fltp> Rc = rat::cmn::Triangle::quad2cart(Rn,Rq);
		const arma::Mat<rat::fltp> dN = rat::cmn::Triangle::shape_function_derivative(Rq);
		const arma::Mat<rat::fltp> J = rat::cmn::Triangle::shape_function_jacobian(Rn,dN);
		const arma::Row<rat::fltp> Jdet = rat::cmn::Triangle::jacobian2determinant(J);
		if(arma::any(Jdet<0.0))rat_throw_line("jacobian determinant is smaller than zero");

		// calculate area
		const rat::fltp area1 = arma::accu(Jdet%wg);
		const rat::fltp area2 = arma::as_scalar(rat::cmn::Triangle::calc_area(Rn,n));

		// check result
		if(std::abs(area1 - area2)/std::abs(area2)>tol)
			rat_throw_line("calculated area is out of tolerance");
	}

	// walk over number of gauss points
	for(arma::uword i=1;i<=8;i++){
		std::cout<<i<<std::endl;

		// create gauss points
		const arma::Mat<rat::fltp> gp = rat::cmn::Triangle::create_gauss_points(i);
		const arma::Mat<rat::fltp> Rq = gp.rows(0,1);
		const arma::Row<rat::fltp> wg = gp.row(2);
		if(std::abs(arma::accu(wg)-1.0/2)>1e-7)rat_throw_line("weights do not sum up to 1/2");

		// const arma::Mat<rat::fltp> Rc = rat::cmn::Triangle::quad2cart(Rn,Rq);
		const arma::Mat<rat::fltp> dN = rat::cmn::Triangle::shape_function_derivative(Rq);
		const arma::Mat<rat::fltp> J = rat::cmn::Triangle::shape_function_jacobian(Rn,dN);
		const arma::Row<rat::fltp> Jdet = rat::cmn::Triangle::jacobian2determinant(J);
		if(arma::any(Jdet<0.0))rat_throw_line("jacobian determinant is smaller than zero");

		// calculate area
		const rat::fltp area1 = arma::accu(Jdet%wg);
		const rat::fltp area2 = arma::as_scalar(rat::cmn::Triangle::calc_area(Rn,n));

		// check result
		if(std::abs(area1 - area2)/std::abs(area2)>tol)
			rat_throw_line("calculated area is out of tolerance");
	}

	// return
	return 0;
}