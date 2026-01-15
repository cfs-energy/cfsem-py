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

#include "stmat.hh"
#include "common/extra.hh"

// main
int main(){
	arma::uword num_targets = 2;
	arma::uword num_exp = 5;
	arma::Mat<double> dR(3,num_targets,arma::fill::zeros);
	
	arma::Mat<std::complex<double> > Lp(Extra::polesize(num_exp),3,arma::fill::randu);

	//Lp(1,2) = std::complex<double>(1,0);
	//Lp(2,2) = std::complex<double>(0,1);
	//Lp(3,2) = std::complex<double>(0,1);
	
	//Lp(1,1) = std::complex<double>(1,1);
	//Lp(2,1) = std::complex<double>(1,0);
	//Lp(3,1) = std::complex<double>(1,0);
	
	//Lp(1,0) = std::complex<double>(1,0);
	//Lp(2,0) = std::complex<double>(1,0);
	//Lp(3,0) = std::complex<double>(1,0);

	ShStMat_Lp2Ta_HPr matrix = StMat_Lp2Ta_H::create();
	matrix->set_num_exp(num_exp);
	matrix->calc_matrix2(dR);

	arma::Mat<std::complex<double> > M = matrix->get_matrix();



	// get real part and return transpose
	arma::Mat<double> dLLp = arma::real(M*Lp);

	// cross product matrix
	arma::Mat<double>::fixed<3,9> Mcr = StMat_Lp2Ta_H::get_cross_product_matrix();

	// calculate cross product
	arma::Mat<double> Bgood = Mcr*arma::reshape(dLLp,num_targets,9).st()/(4*arma::datum::pi);


	// Bother
	matrix->calc_matrix(dR);
	arma::Mat<double> Bnew = matrix->apply(Lp);





	std::cout<<arma::join_horiz(Bgood,Bnew)<<std::endl;

	// return
	return 0;
}