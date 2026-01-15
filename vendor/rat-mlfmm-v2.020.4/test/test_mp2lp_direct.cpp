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

// specific headers
#include "rat/common/error.hh"
#include "fmmmat.hh"
#include "rat/common/extra.hh"

// main
int main(){
	// settings
	const int num_exp = 3;
	const arma::uword num_trans = 2;
	const arma::uword num_dim = 3;

	// set random number generator seed
	arma::arma_rng::set_seed(1002);

	// calcultaew
	const arma::uword polesize = rat::fmm::Extra::polesize(num_exp);

	// create a random set of multipoles
	const arma::Mat<std::complex<rat::fltp> > Mp(polesize,num_trans*num_dim,arma::fill::randu);

	// create a random set of translations
	arma::Mat<rat::fltp> dR(3,num_trans,arma::fill::randu); dR-=0.5; dR*=0.1;

	// multipole to localpole translation without matrix
	const arma::Mat<std::complex<rat::fltp> > Lp1 = rat::fmm::FmmMat_Mp2Lp::calc_direct(Mp, dR, num_exp, num_dim);

	// allocate localpole
	arma::Mat<std::complex<rat::fltp> > Lp2(polesize,num_trans*num_dim);

	// now with matrices
	rat::fmm::FmmMat_Mp2Lp M(num_exp,dR);
	for(arma::uword k=0;k<num_trans;k++)
	 	Lp2.cols(k*num_dim,(k+1)*num_dim-1) = M.apply(k,Mp.cols(k*num_dim,(k+1)*num_dim-1));

	// check output 
	const rat::fltp scale = arma::max(arma::max(arma::abs(Lp1)));
	if(!arma::all(arma::all(arma::abs(arma::real(Lp1-Lp2))/scale<1e-5)))rat_throw_line(
		"multipole to localpole transformation is inconsistent");
	if(!arma::all(arma::all(arma::abs(arma::imag(Lp1-Lp2))/scale<1e-5)))rat_throw_line(
		"multipole to localpole transformation is inconsistent");

	// return
	return 0;
}