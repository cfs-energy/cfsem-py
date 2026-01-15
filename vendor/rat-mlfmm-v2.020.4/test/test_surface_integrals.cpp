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

// general header files
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

// common headers
#include "rat/common/error.hh"
#include "rat/common/elements.hh"

// header files for models
#include "chargedpolyhedron.hh"

// main
int main(){
	// settings
	const rat::fltp width = 0.1;
	const rat::fltp length = 0.08;
	const arma::sword num_gauss = 20;
	const rat::fltp skew = 0.01;
	const rat::fltp tolerance = 1e-6;

	// create nodes for a quadrilateral
	const arma::Mat<rat::fltp>::fixed<3,4> Rn{
		-length/2-skew,-width/2,0.0, 
		-length/2, width/2,0.0, 
		length/2,width/2,0.0, 
		length/2,-width/2,0.0};

	// check
	const arma::Mat<rat::fltp> Rtextra = arma::join_vert(
		Rn.rows(0,1),arma::Row<rat::fltp>(4,arma::fill::value(0.01)));

	// create targets on a sphere
	arma::Mat<rat::fltp> Rnsph; arma::Mat<arma::uword> nsph,ssph;
	rat::cmn::Tetrahedron::create_sphere(Rnsph,nsph,ssph,width,width/10);
	const arma::Mat<rat::fltp> Rt = arma::join_horiz(
		Rnsph.cols(arma::unique(arma::vectorise(ssph))),Rtextra);

	// random points?
	// const arma::Mat<rat::fltp> Rt = arma::randu(3,1e6).eval().each_col() - arma::Col<rat::fltp>::fixed<3>{0.5,0.5,0.5};

	// calcultae
	const arma::Mat<rat::fltp> H1 = rat::fmm::ChargedPolyhedron::calc_magnetic_field(Rn,Rt);
	const arma::Mat<rat::fltp> H2 = rat::fmm::ChargedPolyhedron::calc_magnetic_field_gauss(Rn,Rt,num_gauss);

	// std::cout<<arma::max(arma::abs(arma::vectorise(H1 - H2)))<<std::endl;
	// std::cout<<arma::join_horiz(Rt.t(),H1.t(),H2.t(),H1.t()-H2.t())<<std::endl;
	

	// check for differences
	const rat::fltp error = arma::max(arma::abs(arma::vectorise(H1 - H2)))/arma::max(arma::abs(arma::vectorise(H2)));

	// std::cout<<error<<std::endl;

	// if(error>tolerance)rat_throw_line("error exceeds tolerance.");

	




}