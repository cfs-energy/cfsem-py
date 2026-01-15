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

#include "sparse.hh"
#include "common/extra.hh"

// main
int main(){
	// settings
	arma::uword nc1 = 4;
	arma::uword nc2 = 3;
	arma::uword nr1 = 2;
	arma::uword nr2 = 5;
	double ffill = 0.4;
	bool sort = true;
	
	// create some sparse matrices
	arma::SpMat<std::complex<double> > M1 = arma::sprandu<arma::SpMat<std::complex<double> > >(nr1,nc1,ffill);
	arma::SpMat<std::complex<double> > M2 = arma::sprandu<arma::SpMat<std::complex<double> > >(nr1,nc2,ffill);
	arma::SpMat<std::complex<double> > M3 = arma::sprandu<arma::SpMat<std::complex<double> > >(nr2,nc1,ffill);

	// create field matrix
	arma::field<arma::SpMat<std::complex<double> > > fld(2,3);
	fld(0,0) = M1; fld(0,2) = M2; fld(1,0) = M3;

	// combine matrices
	arma::SpMat<std::complex<double> > M = Sparse::field2mat(fld,sort);

	// display matrices
	Extra::display_mat(M1);
	Extra::display_mat(M2);
	Extra::display_mat(M3);
	Extra::display_mat(M);

	// flag
	bool flag = false;
	
	// check M1
	for(arma::uword i=0;i<nr1;i++){
		for(arma::uword j=0;j<nc1;j++){
			std::complex<double> v1 = M(i,j);
			std::complex<double> v2 = M1(i,j);
			if(v1.real()!=v2.real())flag = true;
			if(v1.imag()!=v2.imag())flag = true;
		}
	}

	// check M2
	for(arma::uword i=0;i<nr1;i++){
		for(arma::uword j=0;j<nc2;j++){
			std::complex<double> v1 = M(i,j+nc1);
			std::complex<double> v2 = M2(i,j);
			if(v1.real()!=v2.real())flag = true;
			if(v1.imag()!=v2.imag())flag = true;
		}
	}

	// check M3
	for(arma::uword i=0;i<nr2;i++){
		for(arma::uword j=0;j<nc1;j++){
			std::complex<double> v1 = M(i+nr1,j);
			std::complex<double> v2 = M3(i,j);
			if(v1.real()!=v2.real())flag = true;
			if(v1.imag()!=v2.imag())flag = true;
		}
	}

	// check empty matrix
	for(arma::uword i=0;i<nr2;i++){
		for(arma::uword j=0;j<nc2;j++){
			std::complex<double> v1 = M(i+nr1,j+nc1);
			if(v1.real()!=0 && flag)flag = true;
			if(v1.imag()!=0 && flag)flag = true;
		}
	}

	// output
	if(flag==false){
		std::printf("all matrices are the same\n");
	}
	else{
		std::printf("all matrices are not the same\n");
	}


	// check flag
	assert(flag==false);
}