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

// solver
#include "chargedpolyhedron.hh"

#include "rat/common/elements.hh"
#include "rat/common/extra.hh"


// main
int main(){
	// settings
	const arma::uword num_gauss = 9llu;
	const arma::uword num_targets = 100llu;
	const rat::fltp tol = RAT_CONST(1e-6);

	// define box corners
	const rat::fltp D = 0.1;
	const rat::fltp dxtrap = 0.05;
	const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2-dxtrap,-D/2-dxtrap,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2,-D/2,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {+D/2,+D/2,0.0}; 
	const arma::Col<rat::fltp>::fixed<3> R3 = {-D/2,+D/2,0.0};
	const arma::Mat<rat::fltp> Rn = arma::join_horiz(R0,R1,R2,R3);

	// create random targets
	const arma::Mat<rat::fltp> Rt = -0.5 + arma::randu(3,num_targets);

	// use charged triangle code to calculate potential
	const arma::Row<rat::fltp> phi1 = rat::fmm::ChargedPolyhedron::calc_scalar_potential(Rn,Rt);

	// create gauss points
	const arma::Mat<rat::fltp> gp = rat::cmn::Quadrilateral::create_gauss_points(num_gauss);
	const arma::Mat<rat::fltp> Rq = gp.rows(0,1); const arma::Row<rat::fltp> wg = gp.row(2);

	// calculate gauss points in carthesian coordinates
	const arma::Mat<rat::fltp> Rc = rat::cmn::Quadrilateral::quad2cart(Rn,Rq);
	const arma::Mat<rat::fltp> dN = rat::cmn::Quadrilateral::shape_function_derivative(Rq);
	const arma::Mat<rat::fltp> J = rat::cmn::Quadrilateral::shape_function_jacobian(Rn,dN);
	const arma::Row<rat::fltp> Jdet = rat::cmn::Quadrilateral::jacobian2determinant(J);

	// calculate integral
	arma::Row<rat::fltp> phi2(num_targets,arma::fill::zeros);
	for(arma::uword i=0;i<num_targets;i++)
		phi2(i) = arma::accu(wg%Jdet/rat::cmn::Extra::vec_norm(Rt.col(i) - Rc.each_col()));

	// show output
	std::cout<<arma::join_vert(phi1,phi2).t()<<std::endl;

	// check
	if(arma::any( (phi2 - phi1)/arma::max(phi2)>tol ))
		rat_throw_line("integral outside of tolerance");

	// return
	return 0;
}
