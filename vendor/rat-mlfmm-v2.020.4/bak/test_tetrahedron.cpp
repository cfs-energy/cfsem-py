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

#include "tetrahedron.hh"
#include "common/gauss.hh"

// main
int main(){
	ShGaussPr gs = Gauss::create(3);
	arma::Row<double> gx = gs->get_get_abscissae();
	arma::Row<double> gw = gs->get_weights();

	arma::Mat<double> Rq(3,gx.n_elem*gx.n_elem*gx.n_elem);
	arma::Row<double> w(gx.n_elem);
	for(arma::uword i=0;i<gx.n_elem;i++){
		for(arma::uword j=0;j<gx.n_elem;j++){
			for(arma::uword k=0;k<gx.n_elem;k++){
				arma::uword idx = i*gx.n_elem*gx.n_elem+j*gx.n_elem+k;
				Rq.col(idx) = arma::Col<double>{gx(i),gx(j),gx(k)};
				w.col(idx) = gw(i)*gw(j)*gw(k);
			}
		}
	}

	Rcc = Tetrahedon::quad2cart(Rq
}