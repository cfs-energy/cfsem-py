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
#include "extra.hh"

// main
int main(){
	// settings
	const arma::uword N = 101;
	const rat::fltp tol = RAT_CONST(1e-5);

	// rotation vector
	const arma::Col<rat::fltp>::fixed<3> V = {RAT_CONST(0.1),RAT_CONST(0.5),RAT_CONST(0.2)};
	const arma::Col<rat::fltp> alpha = arma::linspace<arma::Col<rat::fltp> >(
		-arma::Datum<rat::fltp>::pi,arma::Datum<rat::fltp>::pi,N);

	// normalize vector
	const arma::Col<rat::fltp>::fixed<3> Vnorm = 
		V.each_row()/rat::cmn::Extra::vec_norm(V);

	// walk over rotations
	for(arma::uword i=0;i<N;i++){
		// create rotation matrix
		const arma::Mat<rat::fltp>::fixed<3,3> M1 = 
			rat::cmn::Extra::create_rotation_matrix(Vnorm,alpha(i));

		// check if matrix is pure rotation
		if(!rat::cmn::Extra::is_pure_rotation_matrix(M1,tol))
			rat_throw_line("matrix is not rotation matrix");

		// convert back to vector and angle
		const arma::Col<rat::fltp>::fixed<4> V2 = 
			rat::cmn::Extra::rotmatrix2angle(M1);

		// re-create rotation matrix
		arma::Mat<rat::fltp>::fixed<3,3> M2 = 
			rat::cmn::Extra::create_rotation_matrix(V2.rows(0,2),V2(3));

		// compare
		if(!(M1-M2).is_zero(tol))
			rat_throw_line("angle conversion does not result in same matrix");
	}
}

