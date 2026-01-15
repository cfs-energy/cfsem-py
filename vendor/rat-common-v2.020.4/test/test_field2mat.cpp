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
#include "extra.hh"

// main
int main(){
	// set random seed
	arma::arma_rng::set_seed(1001);
	rat::fltp max_val = 100;

	// matrix sizes
	const arma::uword N0 = 7;
	const arma::uword N1 = 6;
	const arma::uword N2 = 9; 
	
	const arma::uword M0 = 6;
	const arma::uword M1 = 10;
	const arma::uword M2 = 12;

	// generate test matrices
	const arma::Mat<arma::uword> M00 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N0,M0,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M01 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N0,M1,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M02 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N0,M2,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M10 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N1,M0,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M11 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N1,M1,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M12 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N1,M2,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M20 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N2,M0,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M21 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N2,M1,arma::fill::randu)*max_val);

	const arma::Mat<arma::uword> M22 = 
		arma::conv_to<arma::Mat<arma::uword> >::from(
		arma::Mat<rat::fltp>(N2,M2,arma::fill::randu)*max_val);

	// make field array to combine
	arma::field<arma::Mat<arma::uword> > M(3,3);

	// put in matrices
	M(0,0) = M00; M(0,1) = M01; M(0,2) = M02;
	M(1,0) = M10; M(1,1) = M11; M(1,2) = M12;
	M(2,0) = M20; M(2,1) = M21; M(2,2) = M22;

	// merge in field2mat
	const arma::Mat<arma::uword> Mch = rat::cmn::Extra::field2mat(M);

	// check if all matrices are present in the new matrix
	if(Mch.n_cols!=M0+M1+M2)rat_throw_line(
		"number of columns is inconsistent");
	if(Mch.n_rows!=N0+N1+N2)rat_throw_line(
		"number of rpws is inconsistent");

	// check first row
	if(!arma::all(arma::all(Mch.submat(0,0,N0-1,M0-1)==M00)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 0 column 0");
	if(!arma::all(arma::all(Mch.submat(0,M0,N0-1,M0+M1-1)==M01)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 0 column 1");
	if(!arma::all(arma::all(Mch.submat(0,M0+M1,N0-1,M0+M1+M2-1)==M02)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 0 column 2");

	// check second row
	if(!arma::all(arma::all(Mch.submat(N0,0,N0+N1-1,M0-1)==M10)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 0");
	if(!arma::all(arma::all(Mch.submat(N0,M0,N0+N1-1,M0+M1-1)==M11)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 1");
	if(!arma::all(arma::all(Mch.submat(N0,M0+M1,N0+N1-1,M0+M1+M2-1)==M12)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 2");
	
	// check third row
	if(!arma::all(arma::all(Mch.submat(N0+N1,0,N0+N1+N2-1,M0-1)==M20)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 0");
	if(!arma::all(arma::all(Mch.submat(N0+N1,M0,N0+N1+N2-1,M0+M1-1)==M21)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 1");
	if(!arma::all(arma::all(Mch.submat(N0+N1,M0+M1,N0+N1+N2-1,M0+M1+M2-1)==M22)))rat_throw_line(
		"output matrix is inconsistent with input matrix for row 1 column 2");
	
	// return
	return 0;
}